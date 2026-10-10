#!/bin/sh
set -eu

src=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT HUP INT TERM

cat >"$tmp/probe.c" <<'EOF'
#include <stdio.h>
int dll_probe_main(int argc, char **argv) {
    printf("argc=%d\n", argc);
    for (int i = 0; i < argc; ++i) printf("arg[%d]=%s\n", i, argv[i]);
    return 23;
}
EOF

${CC:-cc} -O2 -fPIC -shared -o "$tmp/probe.so" "$tmp/probe.c"
${CC:-cc} -O2 -o "$tmp/probe" "$src/launcher.c" -ldl

set +e
output=$("$tmp/probe" hello 'two words')
status=$?
set -e
[ "$status" -eq 23 ]
printf '%s\n' "$output" | grep -F 'argc=3' >/dev/null
printf '%s\n' "$output" | grep -F 'arg[1]=hello' >/dev/null
printf '%s\n' "$output" | grep -F 'arg[2]=two words' >/dev/null
printf 'shared launcher test passed\n'
