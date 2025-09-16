# Pytree HDF5 serialisation and lazy-loading

Framework base types and HDF5 serialization utilities.

This module defines the core abstract classes and helpers used for:

- Declarative, type-driven HDF5 serialization/deserialization using PyTables
- Lazy-loading via lightweight virtual nodes that can be fetched on demand
- Config objects that serialize to/from JSON and HDF5
- Interop with pandas and array types (NumPy/JAX)

Key concepts:
- `AbstractHDFSerializable`: protocol for saving/loading objects to HDF5 groups.
- `AbstractVxData`: rich base with automatic field-wise serialization and
  strict equality semantics across heterogeneous content.
- Virtual nodes (`HDFVirtualNode`): placeholders for deferred reads.


## Base class to register type-aware serialisation/deserialisation.

::: ehrax.AbstractModule
    options:
      members: false


---

## Base class for HDF serialisable configuration pytrees

::: ehrax.AbstractConfig
    options:
      members: false

---

## HDF serialisable pytrees with pandas equivalent

### DataFrame equivalent

::: ehrax.AbstractWithDataframeEquivalent
    options:
      members: false


### Series equivalent

::: ehrax.AbstractWithSeriesEquivalent
    options:
      members: false
---

## Generic HDF serialisable pytree

::: ehrax.AbstractVxData
    options:
      members: false
---

## Placeholder node for deferred loaded nodes

::: ehrax.HDFVirtualNode
    options:
      members: false

---

## Materialising lazy-loaded nodes

::: ehrax.fetch_at
---

::: ehrax.fetch_one_level_at

---

::: ehrax.fetch_all
