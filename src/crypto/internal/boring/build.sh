#!/bin/bash
# Copyright 2020 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e
id
date
export LANG=C
unset LANGUAGE

# Build BoringCrypto libcrypto.a.
# Following https://csrc.nist.gov/CSRC/media/projects/cryptographic-module-validation-program/documents/security-policies/140sp3678.pdf page 19.

tar xJf boringssl-*z

# Go requires -fPIC for linux/amd64 cgo builds.
# Setting -fPIC only affects the compilation of the non-module code in libcrypto.a,
# because the FIPS module itself is already built with -fPIC.
echo '#!/bin/bash
exec clang-7 -fPIC "$@"
' >/usr/local/bin/clang
echo '#!/bin/bash
exec clang++-7 -fPIC "$@"
' >/usr/local/bin/clang++
chmod +x /usr/local/bin/clang /usr/local/bin/clang++

# The BoringSSL tests use Go, and cgo would look for gcc.
export CGO_ENABLED=0

# Verbatim instructions from BoringCrypto build docs.
printf "set(CMAKE_C_COMPILER \"clang\")\nset(CMAKE_CXX_COMPILER \"clang++\")\n" >${HOME}/toolchain
cd boringssl
mkdir build && cd build && cmake -GNinja -DCMAKE_TOOLCHAIN_FILE=${HOME}/toolchain -DFIPS=1 -DCMAKE_BUILD_TYPE=Release ..
ninja
ninja run_tests

cd ../..

if [ "$(./boringssl/build/tool/bssl isfips)" != 1 ]; then
	echo "NOT FIPS"
	exit 2
fi

# Build and run test C++ program to make sure goboringcrypto.h matches openssl/*.h.
# Also collect list of checked symbols in syms.txt
set -x
set -e
cd godriver
cat >goboringcrypto.cc <<'EOF'
#include <cassert>
#include "goboringcrypto0.h"
#include "goboringcrypto1.h"
#define check_size(t) if(sizeof(t) != sizeof(GO_ ## t)) {printf("sizeof(" #t ")=%d, but sizeof(GO_" #t ")=%d\n", (int)sizeof(t), (int)sizeof(GO_ ## t)); ret=1;}
#define check_func(f) { auto x = f; x = _goboringcrypto_ ## f ; }
#define check_value(n, v) if(n != v) {printf(#n "=%d, but goboringcrypto.h defines it as %d\n", (int)n, (int)v); ret=1;}
int main() {
int ret = 0;
#include "goboringcrypto.x"
return ret;
}
EOF

awk '
BEGIN {
	exitcode = 0
}

# Ignore comments, #includes, blank lines.
/^\/\// || /^#/ || NF == 0 { next }

# Ignore unchecked declarations.
/\/\*unchecked/ { next }

# Check enum values.
!enum && $1 == "enum" && $NF == "{" {
	enum = 1
	next
}
enum && $1 == "};" {
	enum = 0
	next
}
enum && NF == 3 && $2 == "=" {
	name = $1
	sub(/^GO_/, "", name)
	val = $3
	sub(/,$/, "", val)
	print "check_value(" name ", " val ")" > "goboringcrypto.x"
	next
}
enum {
	print FILENAME ":" NR ": unexpected line in enum: " $0 > "/dev/stderr"
	exitcode = 1
	next
}

# Check struct sizes.
/^typedef struct / && $NF ~ /^GO_/ {
	name = $NF
	sub(/^GO_/, "", name)
	sub(/;$/, "", name)
	print "check_size(" name ")" > "goboringcrypto.x"
	next
}

# Check function prototypes.
/^(const )?[^ ]+ \**_goboringcrypto_.*\(/ {
	name = $2
	if($1 == "const")
		name = $3
	sub(/^\**_goboringcrypto_/, "", name)
	sub(/\(.*/, "", name)
	print "check_func(" name ")" > "goboringcrypto.x"
	print name > "syms.txt"
	next
}

{
	print FILENAME ":" NR ": unexpected line: " $0 > "/dev/stderr"
	exitcode = 1
}

END {
	exit exitcode
}
' goboringcrypto.h

cat goboringcrypto.h | awk '
	/^\/\/ #include/ {sub(/\/\//, ""); print > "goboringcrypto0.h"; next}
	/typedef struct|enum ([a-z_]+ )?{|^[ \t]/ {print;next}
	{gsub(/GO_/, ""); gsub(/enum go_/, "enum "); print}
' >goboringcrypto1.h
clang++ -std=c++11 -fPIC -I../boringssl/include -O2 -o a.out  goboringcrypto.cc
./a.out || exit 2

# Prepare copy of libcrypto.a with only the checked functions renamed and exported.
# All other symbols are left alone and hidden.
echo BORINGSSL_bcm_power_on_self_test >>syms.txt
awk '{print "_goboringcrypto_" $0 }' syms.txt >globals.txt
awk '{print $0 " _goboringcrypto_" $0 }' syms.txt >renames.txt
objcopy --globalize-symbol=BORINGSSL_bcm_power_on_self_test ../boringssl/build/crypto/libcrypto.a libcrypto.a

# clang implements u128 % u128 -> u128 by calling __umodti3,
# which is in libgcc. To make the result self-contained even if linking
# against a different compiler version, link our own __umodti3 into the syso.
# This one is specialized so it only expects divisors below 2^64,
# which is all BoringCrypto uses. (Otherwise it will seg fault.)
cat >umod.s <<'EOF'
# tu_int __umodti3(tu_int x, tu_int y)
# x is rsi:rdi, y is rcx:rdx, return result is rdx:rax.
.globl __umodti3
__umodti3:
	# specialized to u128 % u64, so verify that
	test %rcx,%rcx
	jne 1f

	# save divisor
	movq %rdx, %r8

	# reduce top 64 bits mod divisor
	movq %rsi, %rax
	xorl %edx, %edx
	divq %r8

	# reduce full 128-bit mod divisor
	# quotient fits in 64 bits because top 64 bits have been reduced < divisor.
	# (even though we only care about the remainder, divq also computes
	# the quotient, and it will trap if the quotient is too large.)
	movq %rdi, %rax
	divq %r8

	# expand remainder to 128 for return
	movq %rdx, %rax
	xorl %edx, %edx
	ret

1:
	# crash - only want 64-bit divisor
	xorl %ecx, %ecx
	movl %ecx, 0(%ecx)
	jmp 1b

.section .note.GNU-stack,"",@progbits
EOF
clang -c -o umod.o umod.s

ld -r -nostdlib --whole-archive -o goboringcrypto.o libcrypto.a umod.o
echo __umodti3 _goboringcrypto___umodti3 >>renames.txt
objcopy --remove-section=.llvm_addrsig goboringcrypto.o goboringcrypto1.o # b/179161016
objcopy --redefine-syms=renames.txt goboringcrypto1.o goboringcrypto2.o
objcopy --keep-global-symbols=globals.txt goboringcrypto2.o goboringcrypto_linux_amd64.syso

# Done!
ls -l goboringcrypto_linux_amd64.syso
sha256sum goboringcrypto_linux_amd64.syso
