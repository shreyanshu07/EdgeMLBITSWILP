# LogiEdge — OTA Layer-Cache Bandwidth Saving (Task D2)

Run `demo/ota_layer_cache_demo.sh` on a Docker-enabled machine, then replace
the illustrative numbers below with your own measured values before
submitting the Final Report (Section 3).

## Method

Without Docker layer caching, pushing an updated model to the fleet means
every truck's container runtime must pull the **entire image** again (base
OS + Python + all pip dependencies + app code + model), because a naive
registry re-push with no layer reuse ships the full flattened image.

With Docker's content-addressable layer cache (and a registry that supports
layer-level pulls, e.g. a private Docker Registry / Harbor), each layer is
identified by a content hash. Because the Dockerfile places the expensive
`pip install` layer **before** the model-file layer (see `inference/Dockerfile`
comments), a model-only change invalidates only the final, tiny layer — every
truck's container runtime already has the earlier layers cached locally and
only needs to pull the new model layer.

```
Bandwidth WITHOUT layer caching = Full image size x 85 trucks
Bandwidth WITH layer caching    = Changed-layer size x 85 trucks
Savings                          = (Full image size - Changed-layer size) x 85 trucks
```

## Illustrative calculation (replace with your measured `docker images` size)

Using representative figures for a `python:3.11-slim` + `tensorflow-cpu`
image (~450 MB full image, a typical size for this dependency set) and the
production model-file size given in Task E3 (280 KB INT8 TFLite):

```
Full image size          ~= 450 MB
Changed (model) layer    ~= 280 KB = 0.273 MB

Without caching: 450 MB    x 85 trucks = 38,250 MB  ~= 37.35 GB  per update
With caching:      0.273 MB x 85 trucks =    23.2 MB            per update

Savings per update cycle ~= 37.33 GB  (>= 99.9% reduction in update payload)
At M2M cost Rs 0.10/MB:
  Without caching: Rs 3,825 per update cycle (85 trucks)
  With caching:    Rs 2.32  per update cycle (85 trucks)
```

**Action required before submission:** run `docker images logibridge-inference
--format "table {{.Tag}}\t{{.Size}}"` after both builds in the demo script and
substitute the real measured image size and real model-file size above. The
qualitative conclusion (>99% reduction) will hold regardless of the exact
image size, because a full ML-framework image is always 3-4 orders of
magnitude larger than a single quantised model file.
