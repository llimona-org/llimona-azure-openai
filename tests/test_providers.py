"""Unit tests for the Azure OpenAI ``Provider``, ``Responses`` and ``Models`` classes.

The only real external effect in these classes is the ``AsyncAzureOpenAI``
driver, which would otherwise issue HTTP calls to Azure. That driver (and its
``responses`` / ``models`` resources) is the single thing mocked here, using
``unittest.mock``; everything else exercises real ``llimona`` request/response
models so the tests stay close to actual behaviour.

Assumptions (per the repo test instructions):
- ``Context`` only ever has its ``.request`` attribute read by the code under
  test, so it is stubbed with a ``Mock`` exposing ``.request`` rather than
  building a full ``Llimona`` app.
- ``Responses.create`` contains a ``breakpoint()`` on its error path; the error
  path test neutralises ``builtins.breakpoint`` so it does not enter a debugger.
"""

from collections.abc import AsyncIterable
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from llimona.interfaces.openai import Models as BaseModels
from llimona.interfaces.openai import Responses as BaseResponses
from llimona.interfaces.openai.models.api_models import ListModelsRequest, Model, ModelRequest
from llimona.interfaces.openai.models.api_responses import CreateResponse, DeleteResponse, RetrieveResponse
from llimona.interfaces.openai.models.content import ItemReferenceParam
from llimona.interfaces.openai.models.events import ResponseStreamEventTypeAdapter
from llimona.interfaces.openai.models.response import Response
from llimona.providers import ProviderModelDesc, ProviderServiceDesc
from llimona.utils import AsyncIterableMapper
from openai import AsyncAzureOpenAI, AsyncStream

from llimona_azure_openai.providers import (
    Credentials,
    Models,
    Provider,
    ProviderDesc,
    ProviderModel,
    Responses,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_context(request):
    """A ``Context`` stand-in: the services only read ``.request``."""
    return Mock(request=request)


def raw_response(**overrides):
    """A mock OpenAI SDK response whose ``model_dump`` yields a valid payload."""
    data = {
        'id': 'resp_123',
        'status': 'completed',
        'created_at': 1700000000,
        'model': 'gpt-4o',
        'output': [],
    }
    data.update(overrides)
    mock = MagicMock(name='OpenAIResponse')
    mock.model_dump.return_value = data
    return mock


async def aiter_items(items):
    """Wrap a list in an async iterator (what ``driver.models.list()`` returns)."""
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def provider_desc() -> ProviderDesc:
    return ProviderDesc(
        name='azure',
        owner_id='owner-1',
        base_url='https://example.openai.azure.com',
        credentials=Credentials(apiKey='secret-key'),
        services=[
            {'type': 'openai_responses'},
            {'type': 'openai_models'},
        ],
        models=[
            {'name': 'gpt-4o', 'allowedServices': ['openai_responses']},
            {'name': 'open-model'},  # no restrictions -> allowed for any service
            {'name': 'models-only', 'allowedServices': ['openai_models']},
        ],
    )


@pytest.fixture
def provider(provider_desc: ProviderDesc) -> Provider:
    return Provider(provider_desc)


@pytest.fixture
def responses_service(provider: Provider) -> Responses:
    return Responses(provider=provider, service=ProviderServiceDesc(type='openai_responses'))


@pytest.fixture
def models_service(provider: Provider) -> Models:
    return Models(provider=provider, service=ProviderServiceDesc(type='openai_models'))


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class ProviderTests:
    def test_init_builds_azure_driver_from_desc(self, provider: Provider) -> None:
        assert isinstance(provider.driver, AsyncAzureOpenAI)
        assert str(provider.driver.base_url).startswith('https://example.openai.azure.com')
        assert provider.driver.api_key == 'secret-key'
        assert provider.driver.max_retries == 0

    def test_provider_property_exposes_desc(self, provider: Provider, provider_desc: ProviderDesc) -> None:
        assert provider.provider is provider_desc

    def test_build_service_returns_responses(self, provider: Provider) -> None:
        service = provider._build_service(ProviderServiceDesc(type='openai_responses'))
        assert isinstance(service, Responses)
        assert isinstance(service, BaseResponses)
        assert service.provider is provider

    def test_build_service_returns_models(self, provider: Provider) -> None:
        service = provider._build_service(ProviderServiceDesc(type='openai_models'))
        assert isinstance(service, Models)
        assert isinstance(service, BaseModels)
        assert service.provider is provider

    def test_build_service_unknown_type_raises(self, provider: Provider) -> None:
        with pytest.raises(ValueError, match='no available for provider'):
            provider._build_service(ProviderServiceDesc(type='unsupported'))

    def test_build_model_returns_provider_model(self, provider: Provider) -> None:
        model = provider._build_model(ProviderModelDesc(name='gpt-4o'))
        assert isinstance(model, ProviderModel)
        assert model.desc.name == 'gpt-4o'
        assert model.provider is provider

    def test_getattr_lazily_builds_and_caches_service(self, provider: Provider) -> None:
        # The base provider resolves declared services by type via ``__getattr__``.
        first = provider.openai_responses
        second = provider.openai_responses
        assert isinstance(first, Responses)
        assert first is second  # cached on the provider


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class ResponsesTests:
    # --- mapping helpers ---------------------------------------------------

    def test_map_response_validates_into_response_model(self, responses_service: Responses) -> None:
        result = responses_service._map_response(raw_response(id='resp_map', model='gpt-4o'))
        assert isinstance(result, Response)
        assert result.id == 'resp_map'
        assert result.model == 'gpt-4o'
        assert result.status == 'completed'

    def test_map_raw_response_routes_plain_response_to_mapper(self, responses_service: Responses) -> None:
        result = responses_service._map_raw_response(raw_response())
        assert isinstance(result, Response)

    def test_map_raw_response_routes_stream_to_async_mapper(self, responses_service: Responses) -> None:
        # ``isinstance(_, AsyncStream)`` is the discriminator; a spec'd mock satisfies it.
        stream = MagicMock(spec=AsyncStream)
        result = responses_service._map_raw_response(stream)
        assert isinstance(result, AsyncIterableMapper)
        assert isinstance(result, AsyncIterable)

    def test_map_stream_wraps_in_async_iterable_mapper(self, responses_service: Responses) -> None:
        result = responses_service._map_stream(MagicMock(spec=AsyncStream))
        assert isinstance(result, AsyncIterableMapper)

    def test_map_stream_event_without_nested_response_revalidates(self, responses_service: Responses) -> None:
        event = ResponseStreamEventTypeAdapter.validate_python(
            {
                'type': 'response.output_text.delta',
                'content_index': 0,
                'delta': 'hello',
                'item_id': 'item_1',
                'output_index': 0,
                'sequence_number': 1,
            }
        )
        result = responses_service._map_stream_event(event)
        assert result.type == 'response.output_text.delta'
        assert result.delta == 'hello'

    def test_map_stream_event_with_nested_response_maps_it(self, responses_service: Responses) -> None:
        event = ResponseStreamEventTypeAdapter.validate_python(
            {
                'type': 'response.completed',
                'sequence_number': 2,
                'response': {
                    'id': 'resp_evt',
                    'status': 'completed',
                    'created_at': 1700000000,
                    'model': 'gpt-4o',
                    'output': [],
                },
            }
        )
        result = responses_service._map_stream_event(event)
        assert result.type == 'response.completed'
        assert isinstance(result.response, Response)
        assert result.response.id == 'resp_evt'

    # --- create ------------------------------------------------------------

    async def test_create_with_string_input_passes_model_and_input(
        self, provider: Provider, responses_service: Responses
    ) -> None:
        provider.driver.responses.create = AsyncMock(return_value=raw_response(id='created'))
        ctx = make_context(CreateResponse(model='gpt-4o', input='hello world'))

        result = await responses_service.create(ctx)

        assert isinstance(result, Response)
        assert result.id == 'created'
        kwargs = provider.driver.responses.create.call_args.kwargs
        assert kwargs['model'] == 'gpt-4o'
        assert kwargs['input'] == 'hello world'

    async def test_create_with_list_input_dumps_items(self, provider: Provider, responses_service: Responses) -> None:
        provider.driver.responses.create = AsyncMock(return_value=raw_response())
        ctx = make_context(CreateResponse(model='gpt-4o', input=[ItemReferenceParam(item_id='ref-1')]))

        await responses_service.create(ctx)

        kwargs = provider.driver.responses.create.call_args.kwargs
        assert isinstance(kwargs['input'], list)
        assert kwargs['input'][0]['item_id'] == 'ref-1'

    async def test_create_forwards_explicitly_set_optional_params(
        self, provider: Provider, responses_service: Responses
    ) -> None:
        provider.driver.responses.create = AsyncMock(return_value=raw_response())
        ctx = make_context(
            CreateResponse(
                model='gpt-4o',
                input='hi',
                instructions='be brief',
                max_output_tokens=64,
                temperature=0.5,
            )
        )

        await responses_service.create(ctx)

        kwargs = provider.driver.responses.create.call_args.kwargs
        assert kwargs['instructions'] == 'be brief'
        assert kwargs['temperature'] == pytest.approx(0.5)
        # NOTE: optional params are dumped with ``by_alias=True``, and llimona's
        # ``BaseModel`` uses a camelCase alias generator -- so multi-word params
        # reach the driver camelCased (``maxOutputTokens``), NOT as the
        # snake_case kwargs the OpenAI SDK expects. This asserts the *current*
        # behaviour; it likely indicates a bug worth fixing in providers.py.
        assert kwargs['maxOutputTokens'] == 64
        assert 'max_output_tokens' not in kwargs

    async def test_create_omits_unset_optional_params(self, provider: Provider, responses_service: Responses) -> None:
        provider.driver.responses.create = AsyncMock(return_value=raw_response())
        ctx = make_context(CreateResponse(model='gpt-4o', input='hi'))

        await responses_service.create(ctx)

        kwargs = provider.driver.responses.create.call_args.kwargs
        # Only model + input were explicitly set; defaulted optionals must not leak.
        assert set(kwargs) == {'model', 'input'}

    async def test_create_rejects_model_not_allowed_for_service(
        self, provider: Provider, responses_service: Responses
    ) -> None:
        provider.driver.responses.create = AsyncMock(return_value=raw_response())
        # 'models-only' is allowed only for openai_models, not openai_responses.
        ctx = make_context(CreateResponse(model='models-only', input='hi'))

        with pytest.raises(ValueError, match='is not allowed for the provider'):
            await responses_service.create(ctx)

        provider.driver.responses.create.assert_not_called()

    async def test_create_allows_model_without_service_restrictions(
        self, provider: Provider, responses_service: Responses
    ) -> None:
        provider.driver.responses.create = AsyncMock(return_value=raw_response())
        ctx = make_context(CreateResponse(model='open-model', input='hi'))

        result = await responses_service.create(ctx)

        assert isinstance(result, Response)

    async def test_create_propagates_driver_errors(
        self, provider: Provider, responses_service: Responses, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The error path calls ``breakpoint()``; neutralise it so the test does
        # not drop into a debugger.
        monkeypatch.setattr('builtins.breakpoint', lambda *a, **k: None)
        provider.driver.responses.create = AsyncMock(side_effect=RuntimeError('boom'))
        ctx = make_context(CreateResponse(model='gpt-4o', input='hi'))

        with pytest.raises(RuntimeError, match='boom'):
            await responses_service.create(ctx)

    # --- retrieve ----------------------------------------------------------

    async def test_retrieve_without_include_omits_it(self, provider: Provider, responses_service: Responses) -> None:
        provider.driver.responses.retrieve = AsyncMock(return_value=raw_response(id='retrieved'))
        ctx = make_context(RetrieveResponse(response_id='resp_abc'))

        result = await responses_service.retrieve(ctx)

        assert isinstance(result, Response)
        assert result.id == 'retrieved'
        kwargs = provider.driver.responses.retrieve.call_args.kwargs
        assert kwargs['response_id'] == 'resp_abc'
        assert 'include' not in kwargs

    async def test_retrieve_with_include_forwards_it(self, provider: Provider, responses_service: Responses) -> None:
        provider.driver.responses.retrieve = AsyncMock(return_value=raw_response())
        ctx = make_context(RetrieveResponse(response_id='resp_abc', include=['file_search_call.results']))

        await responses_service.retrieve(ctx)

        kwargs = provider.driver.responses.retrieve.call_args.kwargs
        assert kwargs['include'] == ['file_search_call.results']

    # --- cancel ------------------------------------------------------------

    async def test_cancel_calls_driver_and_maps_response(
        self, provider: Provider, responses_service: Responses
    ) -> None:
        provider.driver.responses.cancel = AsyncMock(return_value=raw_response(id='cancelled', status='incomplete'))
        ctx = make_context(DeleteResponse(response_id='resp_xyz'))

        result = await responses_service.cancel(ctx)

        assert isinstance(result, Response)
        assert result.id == 'cancelled'
        assert provider.driver.responses.cancel.call_args.kwargs['response_id'] == 'resp_xyz'


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ModelsTests:
    async def test_list_yields_mapped_models(self, provider: Provider, models_service: Models) -> None:
        items = [
            Mock(id='gpt-4o', owned_by='openai', created=1700000000),
            Mock(id='gpt-4o-mini', owned_by='org', created=1700000001),
        ]
        provider.driver.models.list = MagicMock(return_value=aiter_items(items))
        ctx = make_context(ListModelsRequest())

        models = [m async for m in models_service.list(ctx)]

        assert [m.id for m in models] == ['gpt-4o', 'gpt-4o-mini']
        assert all(isinstance(m, Model) for m in models)
        assert models[0].owned_by == 'openai'
        assert models[0].created == 1700000000

    async def test_list_with_no_models_yields_nothing(self, provider: Provider, models_service: Models) -> None:
        provider.driver.models.list = MagicMock(return_value=aiter_items([]))
        ctx = make_context(ListModelsRequest())

        models = [m async for m in models_service.list(ctx)]

        assert models == []

    async def test_retrieve_maps_driver_model(self, provider: Provider, models_service: Models) -> None:
        provider.driver.models.retrieve = AsyncMock(
            return_value={'id': 'gpt-4o', 'created': 1700000000, 'owned_by': 'openai'}
        )
        ctx = make_context(ModelRequest(model_id='gpt-4o'))

        model = await models_service.retrieve(ctx)

        assert isinstance(model, Model)
        assert model.id == 'gpt-4o'
        assert provider.driver.models.retrieve.call_args.kwargs['model'] == 'gpt-4o'

    async def test_delete_returns_deleted_flag_true(self, provider: Provider, models_service: Models) -> None:
        provider.driver.models.delete = AsyncMock(return_value=Mock(deleted=True))
        ctx = make_context(ModelRequest(model_id='ft:gpt-4o'))

        result = await models_service.delete(ctx)

        assert result is True
        assert provider.driver.models.delete.call_args.kwargs['model'] == 'ft:gpt-4o'

    async def test_delete_returns_deleted_flag_false(self, provider: Provider, models_service: Models) -> None:
        provider.driver.models.delete = AsyncMock(return_value=Mock(deleted=False))
        ctx = make_context(ModelRequest(model_id='gpt-4o'))

        result = await models_service.delete(ctx)

        assert result is False
