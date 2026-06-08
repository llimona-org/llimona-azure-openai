"""Unit tests for ``AzureOpenAIAddon`` and the ``addon`` factory function.

These exercise real ``llimona`` registry machinery rather than mocking it: a
``ComponentRegistry`` instance is cheap to build and lets us assert the actual
registration outcome instead of merely that a method was called.
"""

import pytest
from llimona.addons import AddonMetadata
from llimona.registries import ComponentRegistry

from llimona_azure_openai import AzureOpenAIAddon, addon
from llimona_azure_openai.providers import Provider, ProviderDesc


@pytest.fixture
def registry() -> ComponentRegistry:
    return ComponentRegistry(name='test-providers')


# ---------------------------------------------------------------------------
# AzureOpenAIAddon
# ---------------------------------------------------------------------------


class AzureOpenAIAddonTests:
    def test_is_addon_metadata(self) -> None:
        assert isinstance(AzureOpenAIAddon(), AddonMetadata)

    def test_metadata_values(self) -> None:
        instance = AzureOpenAIAddon()
        assert instance.name == 'azure_openai'
        assert instance.display_name == 'Azure OpenAI Addon'
        assert instance.description == 'An addon to support Azure OpenAI as a provider in Llimona.'

    def test_register_providers_registers_provider_under_azure_openai_type(
        self, registry: ComponentRegistry
    ) -> None:
        AzureOpenAIAddon().register_providers(registry)

        # 'azure_openai' is the default value of ``ProviderDesc.type``, which the
        # registry uses as the lookup key.
        assert registry.get_description_class('azure_openai') is ProviderDesc
        assert registry.get_component_class('azure_openai') is Provider

    def test_register_providers_is_idempotent(self, registry: ComponentRegistry) -> None:
        addon_instance = AzureOpenAIAddon()
        addon_instance.register_providers(registry)
        addon_instance.register_providers(registry)

        assert registry.get_component_class('azure_openai') is Provider

    def test_register_providers_unknown_type_raises_keyerror(self, registry: ComponentRegistry) -> None:
        AzureOpenAIAddon().register_providers(registry)

        with pytest.raises(KeyError):
            registry.get_component_class('not_registered')


# ---------------------------------------------------------------------------
# addon()
# ---------------------------------------------------------------------------


class AddonFunctionTests:
    def test_returns_azure_openai_addon_instance(self) -> None:
        result = addon()
        assert isinstance(result, AzureOpenAIAddon)
        assert isinstance(result, AddonMetadata)

    def test_is_cached_and_returns_same_instance(self) -> None:
        # ``addon`` is wrapped in ``lru_cache(maxsize=1)``.
        assert addon() is addon()
