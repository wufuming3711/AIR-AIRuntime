#!/bin/bash
# FFMPEG_VERSION="release/7.1"
#BuildForWindows="1"
#rm -fr "./ffmpeg_src"
#git clone https://github.com/FFmpeg/FFmpeg.git ffmpeg_src
rm -fr "./ffmpeg_bld"
mkdir "./ffmpeg_bld"
rm -fr "./ffmpeg"
SOURCE_PATH=$(pwd)
cd ffmpeg_src || exit
# git checkout $FFMPEG_VERSION
CONFIGURE_OPTIONS=(
"--prefix=${SOURCE_PATH}/ffmpeg"
"--disable-all"
# 这都是CPU优化,不是特别古典的CPU都支持,除了第一个
"--enable-avx512"
"--enable-asm"
"--enable-mmx"
"--enable-mmxext"
"--enable-sse"
"--enable-sse2"
"--enable-sse3"
"--enable-ssse3"
"--enable-sse4"
"--enable-sse42"
"--enable-avx"
"--enable-avx2"
"--enable-aesni"
"--enable-inline-asm"
"--enable-x86asm"
# 启用硬件加速支持
#"--enable-hwaccels"
"--disable-vdpau"
"--enable-gpl"
"--enable-version3"
"--enable-nonfree"
#"--disable-shared"
"--enable-static"

"--enable-avcodec"
"--enable-avformat"

"--enable-swscale"
# "--disable-swscale-alpha"

"--enable-protocol=udp"
"--enable-protocol=tcp"
"--enable-protocol=http"
"--enable-protocol=rtp"
"--enable-protocol=rtmp"

"--enable-demuxer=sdp"
"--enable-demuxer=rtsp"
"--enable-demuxer=rtp"
"--enable-demuxer=flv"

"--enable-parser=h264"
"--enable-parser=hevc"
"--enable-decoder=h264"
"--enable-decoder=hevc"

"--disable-pthreads"
)
if [ -n "$BuildForWindows" ]; then
    CONFIGURE_OPTIONS+=("--target-os=win64")
    CONFIGURE_OPTIONS+=("--arch=x86_64")
    CONFIGURE_OPTIONS+=("--toolchain=msvc")
    export PATH=$PATH:"/c/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.41.34120/bin/Hostx64/x64"
fi
cd ../ffmpeg_bld || exit
../ffmpeg_src/configure "${CONFIGURE_OPTIONS[@]}"
if [ -n "$BuildForWindows" ]; then
iconv -f GB18030 -t UTF-8 config.h > "1.tmp" && rm config.h
fi
mv 1.tmp config.h
make CFLAGS="-O3 -march=native" CXXFLAGS="-O3 -march=native" -j18
make install

# #!/bin/bash
# # FFMPEG_VERSION="release/7.1"
# #BuildForWindows="1"
# #rm -fr "./ffmpeg_src"
# # git clone https://kkgithub.com/FFmpeg/FFmpeg.git ffmpeg_src
# rm -fr "./ffmpeg_bld"
# mkdir "./ffmpeg_bld"
# # rm -fr "./ffmpeg"
# SOURCE_PATH=$(pwd)
# cd ffmpeg_src || exit
# # git checkout $FFMPEG_VERSION
# CONFIGURE_OPTIONS=(
# "--prefix=${SOURCE_PATH}/ffmpeg"
# "--disable-all"
# # 这都是CPU优化,不是特别古典的CPU都支持,除了第一个
# "--enable-avx512"
# "--enable-asm"
# "--enable-mmx"
# "--enable-mmxext"
# "--enable-sse"
# "--enable-sse2"
# "--enable-sse3"
# "--enable-ssse3"
# "--enable-sse4"
# "--enable-sse42"
# "--enable-avx"
# "--enable-avx2"
# "--enable-aesni"
# "--enable-inline-asm"
# "--enable-x86asm"
# # 启用硬件加速支持
# #"--enable-hwaccels"
# "--disable-vdpau"
# "--enable-gpl"
# "--enable-version3"
# "--enable-nonfree"
# #"--disable-shared"
# "--enable-static"

# "--enable-avcodec"
# "--enable-avformat"

# "--enable-swscale"
# "--disable-swscale-alpha"

# "--enable-protocol=udp"
# "--enable-protocol=tcp"
# "--enable-protocol=http"
# "--enable-protocol=rtp"
# "--enable-protocol=rtmp"

# "--enable-demuxer=sdp"
# "--enable-demuxer=rtsp"
# "--enable-demuxer=rtp"
# "--enable-demuxer=flv"

# "--enable-parser=h264"
# "--enable-parser=hevc"
# "--enable-decoder=h264"
# "--enable-decoder=hevc"
# )
# if [ -n "$BuildForWindows" ]; then
#     CONFIGURE_OPTIONS+=("--target-os=win64")
#     CONFIGURE_OPTIONS+=("--arch=x86_64")
#     CONFIGURE_OPTIONS+=("--toolchain=msvc")
#     export PATH=$PATH:"/c/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.41.34120/bin/Hostx64/x64"
# fi
# cd ../ffmpeg_bld || exit
# ../ffmpeg_src/configure "${CONFIGURE_OPTIONS[@]}"
# if [ -n "$BuildForWindows" ]; then
# iconv -f GB18030 -t UTF-8 config.h > "1.tmp" && rm config.h
# fi
# mv 1.tmp config.h
# # make CFLAGS="-O3 -mtune=haswell" CXXFLAGS="-O3 -mtune=haswell" -j18
# make CFLAGS="-O3 -march=skylake" CXXFLAGS="-O3 -march=skylake" -j18
# make install