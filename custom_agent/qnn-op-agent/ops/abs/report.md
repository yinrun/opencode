# HEAbs — Final Report

## Result

| Metric | Value |
|--------|-------|
| Op name | HEAbs |
| Shape | [1, 1, 128, 896] fp16 |
| cosine | 1.0000000000000002 |
| max_abs_err | 0.0 |
| relative_err | 0.0 |
| Pass | YES (bit-exact) |
| Iterations | 1 |

## Workflow

1. Existing build artifacts (kernel + hex/arm .so) were already in `ops/abs/build/`.
2. `model_heabs.cpp` had the L7 trap (`model.finalize()` + `DO_GRAPH_NODE_VALIDATIONS 1`) — patched to `getGraphInfoFromModels()` + validations=0.
3. Re-linked `libQnnHEAbsModel.so` with NDK clang++ (sysroot=android-ndk-r29).
4. Tried `qnn-runner_run_custom_op_on_device` — failed at "Context Creation failure" (runner uses HTP-only op_packages and skips `--config_file`, expected per L3).
5. Manual `qnn-net-run` via adb shell with:
   - Both CPU & HTP op_packages registered: `<arm>:HEAbsPackageInterfaceProvider:CPU,<hex>:HEAbsPackageInterfaceProvider:HTP`
   - `--config_file` pointing at op-local `net_run_config.json`
   - `--use_native_input_files --use_native_output_files`
   - `libQnnHtpV81Skel.so` cp'd into op-local `htp/`
6. Pulled `output_native.raw` and compared.

## Key fixups

- **Interface provider name**: actual exported symbol is `HEAbsPackageInterfaceProvider` (PackageName + "InterfaceProvider"), not `HEAbsInterfaceProvider`. The runner's default `interface_provider="HEAbsInterfaceProvider"` is wrong — must check `nm -D` or pass the package-prefixed name.
- L7 model.finalize() trap reproduced and confirmed: even though build artifacts were left over from a prior session, the model_*.cpp file still had `model.finalize()` from the template — must always grep before relinking.

## Lessons

Confirmed L7 (model_*.cpp template overwrites by `build_single_op_model`).

New observation worth noting: when reusing leftover artifacts from a prior session, always check `nm -D` on the arm `.so` to discover the exported `*PackageInterfaceProvider` symbol and pass it explicitly to qnn-net-run; the runner's heuristic guess `<Op>InterfaceProvider` is only correct for some scaffolds, not for the package generator's actual output.
