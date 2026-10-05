# Clean MiMo runner image

This recipe builds the Release 8 MiMo runner from the pinned official vLLM
0.30 image. It installs one complete runner wheel and the exact dependency
artifacts listed in `artifact-sha256.txt`. The resulting image contains no
source checkout, test tree, diagnostic overlay, or runtime source mount.

## Stage artifacts

Create `build/mimo_clean/artifacts` and place each file named in
`artifact-sha256.txt` inside it. Copy the manifest into the same directory:

```bash
mkdir -p build/mimo_clean/artifacts
cp build/mimo_clean/artifact-sha256.txt build/mimo_clean/artifacts/
```

The repository does not store compiled wheels or shared objects. Build or
obtain them from their pinned revisions, then verify the complete directory:

```bash
cd build/mimo_clean/artifacts
sha256sum -c artifact-sha256.txt
cd ../../..
```

## Build

```bash
docker build \
  --file build/mimo_clean/Dockerfile \
  --build-arg BASE_IMAGE='vllm/vllm-openai:v0.30.0@sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56' \
  --build-arg VLLM_COMMIT='239ad2b4c8bd53ddd6382a27fc36cfcb8b193a6d' \
  --build-arg RUNNER_VERSION='0.30.0+239ad2b4c' \
  --build-arg EXLLAMAV3_COMMIT='c5d9c657966ffeeaa9353f0cc899f18629da4a13' \
  --build-arg FLASHINFER_COMMIT='dc04f50c9aa3eabcdaa5feb0934edb3d85e9529a' \
  --build-arg SPARKINFER_COMMIT='d4438d490691f79022fdfc8149e1c5f161d15445' \
  --tag cb-vllm-dual-dgx:mimo-release-8 \
  .
```

Run the image on both tensor-parallel ranks without source mounts or package
overlays. Use the MiMo recipe in `recipes/` for the qualified launch contract.
