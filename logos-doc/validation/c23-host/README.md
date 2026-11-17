# C23 host requirement

The completed scripts job in [run 43927955055](https://github.com/gosuda/LoGOs/actions/runs/43927955055/job/133512372234)
failed only `TestScript/logos_cgo_shared`. LoGOs produced `liblogos.so` and
`liblogos.h`; the fixture's external C consumer then failed because the selected
`cc` rejected `-std=c23` and suggested `-std=c2x`.

The tested image was `ubuntu-24.04`, version `20261111.370.1`. Its
[published software manifest](https://github.com/actions/runner-images/blob/ubuntu24/20261111.370/images/ubuntu/Ubuntu2404-Readme.md)
lists GCC 13 as the default and Clang 18.1.3 as an installed compiler. The fixture
clears `CC` and chooses the first available `cc`, `gcc`, or `clang`, so setting
`CC` alone would leave the observed failure unchanged.

The workflow now resolves `clang-18`, records its version and executable hash,
compiles and runs a C23 probe with warnings treated as errors, and exposes that
verified executable as `cc` in a job-local directory on `PATH`. This setup runs
for the scripts and upstream jobs. It changes no system compiler installation
or repository source. A missing compiler or failed C23 probe fails the job.

The original fixture and its consumer assertions remain unchanged. The SHA-256
of `src/cmd/go/testdata/script/logos_cgo_shared.txt` is
`9b347ef0073aeed8ac009e9062b5c44cdaa4324977431dc545e1c6e69233db1b`.
The earlier failed log remains evidence for its recorded host profile.

Repository actionlint and the workflow diff check pass. A hosted rerun of the
unchanged script with this host setup remains required; no local C or Go build
is claimed here.
