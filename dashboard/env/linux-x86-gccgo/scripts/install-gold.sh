set -ex

# gccgo uses the Gold linker from binutils.
export BINUTILS_VERSION=binutils-2.24
mkdir -p binutils-objdir
curl -s http://ftp.gnu.org/gnu/binutils/$BINUTILS_VERSION.tar.gz | tar x --no-same-owner -zv
(cd binutils-objdir && ../$BINUTILS_VERSION/configure --enable-gold --enable-plugins --prefix=/opt/gold && make -sj && make install -sj)

rm -rf binutils*