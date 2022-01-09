# Inference using TBCNN

Tell whether Rust program files are unsafe or not. 

## Usage

```bash
./inference.sh {Rust code file}+
```

## Example

The following commands test whether the functiosn in anyhow are unsafe:

```bash
./inference.sh ../data/safe/anyhow/*.rs
```
outputs:
```
../data/safe/anyhow/lib-new_adhoc.rs: Safe (0.647).
```

```bash
./inference.sh ../data/unsafe/anyhow/*.rs
```
outputs:
```
../data/unsafe/anyhow/error-object_downcast_mut.rs: Unsafe (0.995).
../data/unsafe/anyhow/error-object_downcast.rs: Unsafe (0.995).
../data/unsafe/anyhow/error-object_ref.rs: Unsafe (0.999).
../data/unsafe/anyhow/error-object_drop.rs: Unsafe (1.000).
../data/unsafe/anyhow/error-context_downcast_mut.rs: Unsafe (0.986).
../data/unsafe/anyhow/error-object_drop_front.rs: Unsafe (1.000).
../data/unsafe/anyhow/error-object_boxed.rs: Unsafe (1.000).
../data/unsafe/anyhow/error-context_drop_rest.rs: Unsafe (1.000).
../data/unsafe/anyhow/error-object_mut.rs: Unsafe (1.000).
../data/unsafe/anyhow/error-context_downcast.rs: Unsafe (0.985).
../data/unsafe/anyhow/error-context_chain_downcast_mut.rs: Unsafe (0.777).
../data/unsafe/anyhow/error-context_chain_downcast.rs: Unsafe (0.761).
../data/unsafe/anyhow/error-vtable.rs: Unsafe (0.996).
../data/unsafe/anyhow/error-context_chain_drop_rest.rs: Unsafe (1.000).
```
