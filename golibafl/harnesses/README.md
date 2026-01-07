# Harnesses

## Targets
This directory contains all the targets used to test and evaluate our fuzzer.
The targets include:
- A custom path constraint target
- [prometheus](https://github.com/prometheus/prometheus)
- [caddy](https://github.com/caddyserver/caddy)
- [burntsushi-toml](https://github.com/BurntSushi/toml)

The fuzzed functionality for all non-custom targets was copied from the [oss-fuzz](https://github.com/google/oss-fuzz) repository.

## Concrete example: Caddy 
- Please run all commands from the `crate` root.

**Fuzz**
```bash
# will save fuzzer output in golibafl/output
HARNESS=harnesses/caddy cargo run -r -- fuzz

# will save fuzzer output in golibafl/harnesses/caddy/
HARNESS=harnesses/caddy cargo run -r -- fuzz -o ./harnesses/caddy
```

**Get coverage**
```bash
# get coverage if fuzzer output is in golibafl/output
cargo run -r -- cov -o ./output/queue -f ./harnesses/caddy

# get coverage if fuzzer output is in golibafl/harnesses/caddy
cargo run -r -- cov -o ./harnesses/caddy/queue -f ./harnesses/caddy
```

You can filter for `caddy` and harness specific coverage by providing the package names as coverage filter (`-c`):

```bash
cargo run -r -- cov -o ./output/queue -f ./harnesses/caddy -c fuzz,caddy

cargo run -r -- cov -o ./harnesses/caddy/queue -f ./harnesses/caddy -c fuzz,caddy
```
