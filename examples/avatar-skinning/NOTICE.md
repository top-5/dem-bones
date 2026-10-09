# Third-party notice

`python/avatar_skinning.py` contains an adapted dual-quaternion skinning method
based on NAVER Anny:

- Source: <https://github.com/naver/anny/blob/main/src/anny/skinning/skinning.py>
- Copyright: NAVER Corp.
- Licence: Apache License 2.0

The adaptation adds the shared input contract, validation, broadcasting,
quaternion reference selection, a small differentiable weight-recovery baseline,
and tests. See the upstream repository for its complete licence and notices.

