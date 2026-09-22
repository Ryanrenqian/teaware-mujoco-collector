# Public release policy

Run this before every public push:

```bash
uv run teaware-mj audit-release --root .
```

The audit checks tracked files only. It rejects runtime datasets, checkpoints,
private keys, common cloud tokens, absolute user paths, private network
addresses, and hardware serial ports. It also verifies that the vendored xArm7
license and this notice are present.

Generated datasets remain under the ignored `data/` directory. Review any
dataset intended for publication separately for people, screens, labels,
machine names, camera serial numbers, physical addresses, and proprietary
objects. Publish large approved datasets as versioned release artifacts or in a
dataset registry rather than adding them to the source tree.

The included xHand is a procedural compatibility model. Do not copy the xHand
meshes or SDK from `waic-demo4` into this public repository until the owner has
documented an explicit redistribution license.
