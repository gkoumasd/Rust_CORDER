# Inferencing using Corder [SIGIR'21]

Tell whether Rust program files are unsafe or not. 

## Usage

```bash
./inference.sh {Rust code file}+
```

## Example

The following commands test whether the functions in the `anyhow` crate are safe or unsafe, with the confidence level:

```bash
./inference.sh ../data/safe/anyhow/*.rs
```
outputs:
```
../data/safe/anyhow/lib-new_adhoc.rs:1:safe: Safe (0.647).
```

```bash
./inference.sh ../data/unsafe/anyhow/*.rs
```
outputs:
```
../data/unsafe/anyhow/error-context_drop_rest.rs:2:safe: Unsafe (1.000).
../data/unsafe/anyhow/error-object_boxed.rs:1:safe: Unsafe (1.000).
../data/unsafe/anyhow/error-object_downcast_mut.rs:1:safe: Unsafe (0.995).
../data/unsafe/anyhow/error-object_downcast.rs:1:safe: Unsafe (0.995).
../data/unsafe/anyhow/error-context_chain_drop_rest.rs:1:safe: Unsafe (1.000).
../data/unsafe/anyhow/error-vtable.rs:1:safe: Unsafe (0.996).
../data/unsafe/anyhow/error-context_chain_downcast.rs:1:safe: Unsafe (0.761).
../data/unsafe/anyhow/error-context_chain_downcast_mut.rs:1:safe: Unsafe (0.777).
../data/unsafe/anyhow/error-context_downcast_mut.rs:2:safe: Unsafe (0.986).
../data/unsafe/anyhow/error-context_downcast.rs:2:safe: Unsafe (0.985).
../data/unsafe/anyhow/error-object_drop.rs:1:safe: Unsafe (1.000).
../data/unsafe/anyhow/error-object_drop_front.rs:1:safe: Unsafe (1.000).
../data/unsafe/anyhow/error-object_mut.rs:2:safe: Unsafe (0.999).
../data/unsafe/anyhow/error-object_ref.rs:1:safe: Unsafe (0.999).
```

Note that the ":safe" tag was associated with all the preprocessed functions in the "unsafe" folder in our datasets.

If we test it on the unpreprocessed Rust code, these tags will be derived from the code (only functions starting with "unsafe fn "
are associated with the `:unsafe` tags.

```bash
./inference.sh 
```
outputs:
```
../data/unsafe/anyhow/error-context_drop_rest.rs:2:safe: Unsafe (1.000).
../data/unsafe/anyhow/error-object_boxed.rs:1:safe: Unsafe (1.000).
../data/unsafe/anyhow/error-object_downcast_mut.rs:1:safe: Unsafe (0.995).
../data/unsafe/anyhow/error-object_downcast.rs:1:safe: Unsafe (0.995).
../data/unsafe/anyhow/error-context_chain_drop_rest.rs:1:safe: Unsafe (1.000).
../data/unsafe/anyhow/error-vtable.rs:1:safe: Unsafe (0.996).
../data/unsafe/anyhow/error-context_chain_downcast.rs:1:safe: Unsafe (0.761).
../data/unsafe/anyhow/error-context_chain_downcast_mut.rs:1:safe: Unsafe (0.777).
../data/unsafe/anyhow/error-context_downcast_mut.rs:2:safe: Unsafe (0.986).
../data/unsafe/anyhow/error-context_downcast.rs:2:safe: Unsafe (0.985).
../data/unsafe/anyhow/error-object_drop.rs:1:safe: Unsafe (1.000).
../data/unsafe/anyhow/error-object_drop_front.rs:1:safe: Unsafe (1.000).
../data/unsafe/anyhow/error-object_mut.rs:2:safe: Unsafe (0.999).
../data/unsafe/anyhow/error-object_ref.rs:1:safe: Unsafe (0.999).
```

Note that the ":safe" tag was associated with all the preprocessed functions in the "unsafe" folder in our datasets.

If we test it on the unpreprocessed Rust code, these tags will be derived from the code (only functions starting with "unsafe fn "
are associated with the `:unsafe` tags.

```bash
./inference.sh $HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs
```
outputs:
```
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:399:safe: Safe (0.949).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:28:safe: Safe (0.843).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:922:safe: Unsafe (0.978).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:578:unsafe: Unsafe (0.957).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:757:unsafe: Safe (1.000).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:847:unsafe: Safe (0.709).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:884:unsafe: Safe (0.998).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:894:safe: Safe (0.993).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:349:safe: Safe (0.953).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:529:safe: Safe (0.938).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:555:safe: Unsafe (0.851).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:946:safe: Unsafe (1.000).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:838:safe: Safe (1.000).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:537:safe: Safe (0.887).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:953:safe: Unsafe (1.000).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:376:safe: Safe (0.999).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:518:safe: Safe (0.987).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:939:safe: Unsafe (1.000).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:933:safe: Unsafe (1.000).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:676:safe: Safe (0.820).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:103:safe: Unsafe (0.625).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:128:safe: Safe (0.554).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:596:unsafe: Unsafe (0.505).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:154:safe: Unsafe (1.000).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:491:safe: Safe (1.000).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:773:unsafe: Safe (0.926).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:81:safe: Unsafe (1.000).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:719:unsafe: Unsafe (1.000).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:801:unsafe: Unsafe (0.982).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:824:unsafe: Unsafe (0.950).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:549:safe: Safe (0.992).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:387:safe: Unsafe (0.983).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:73:safe: Unsafe (0.719).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:907:safe: Safe (0.997).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:916:safe: Safe (0.997).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:543:safe: Safe (0.991).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:898:safe: Unsafe (0.989).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:287:safe: Unsafe (0.999).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:407:safe: Safe (1.000).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:616:unsafe: Safe (0.567).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:868:unsafe: Safe (1.000).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:625:unsafe: Unsafe (0.726).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:586:unsafe: Unsafe (0.997).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:661:unsafe: Safe (1.000).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:210:unsafe: Safe (1.000).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:854:unsafe: Safe (0.938).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:477:safe: Safe (1.000).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:634:unsafe: Safe (0.990).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:740:unsafe: Safe (1.000).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:683:unsafe: Unsafe (0.886).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:701:unsafe: Unsafe (0.896).
$HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs:180:safe: Safe (0.565).
```

As you can see, the predictor is not always the same.

Redirect the output to an AWK script, we can compare the labels to tell how well it performs on particular set of functions:
```bash
./inference.sh $HOME/.cargo/registry/src/github.com-1ecc6299db9ec823/anyhow-1.0.40/src/error.rs | awk -f t.awk t.t
```
outputs
```
safe safe
safe safe
safe unsafe
unsafe unsafe
unsafe safe
unsafe safe
unsafe safe
safe safe
safe safe
safe safe
safe unsafe
safe unsafe
safe safe
safe safe
safe unsafe
safe safe
safe safe
safe unsafe
safe unsafe
safe safe
safe unsafe
safe safe
unsafe unsafe
safe unsafe
safe safe
unsafe safe
safe unsafe
unsafe unsafe
unsafe unsafe
unsafe unsafe
safe safe
safe unsafe
safe unsafe
safe safe
safe safe
safe safe
safe unsafe
safe unsafe
safe safe
unsafe safe
unsafe safe
unsafe unsafe
unsafe unsafe
unsafe safe
unsafe safe
unsafe safe
safe safe
unsafe safe
unsafe safe
unsafe unsafe
unsafe unsafe
safe safe
0.538462
```
