# Native fuzzer

## Fuzzing
To fuzz a target, navigate to its directory (e.g., `cd caddy`) and run the following commands:

**Start Fuzzing**

```bash
./fuzz.sh
```
By default, `fuzz.sh` fuzzes with a single worker. To increase this, modify the `-parallel` parameter in the script.

**Generate Coverage Report**

```bash
./get_cov.sh
```
The script produces a `cover.html` file in the target directory.

