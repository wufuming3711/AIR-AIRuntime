#!/bin/bash
GRPC_VERSION="v1.67.1"
rm -fr "./grpc_src"
git clone --recurse-submodules -b $GRPC_VERSION --depth 1 --shallow-submodules https://kkgithub.com/grpc/grpc grpc_src
rm -fr "./grpc_bld"
mkdir "./grpc_bld"
rm -fr "./grpc"
SOURCE_PATH=$(pwd)
cd grpc_src || exit
GRPC_INSTALL_DIR="$SOURCE_PATH/grpc"
export PATH="$GRPC_INSTALL_DIR/bin:$PATH"
cd ../grpc_bld || exit
# cmake -DgRPC_INSTALL=ON \
#       -DgRPC_BUILD_TESTS=OFF \
#       -DCMAKE_INSTALL_PREFIX="$GRPC_INSTALL_DIR" \
#       ../grpc_src
# make -j 18
# make install