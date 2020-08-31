#! /bin/bash
if [ -f .original_env ]; then
  source .original_env
else
  echo "#! /bin/bash"                       >  .original_env
  echo "PATH=$PATH"                         >> .original_env
  echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"   >> .original_env
fi

export LD_LIBRARY_PATH=/home/laura/Documenti/TIROCINIO/pixeltrack-standalone/external/tbb/lib:/usr/local/cuda/lib64:/home/laura/Documenti/TIROCINIO/pixeltrack-standalone/external/cupla/lib:/home/laura/Documenti/TIROCINIO/pixeltrack-standalone/external/kokkos/install/lib:$LD_LIBRARY_PATH
export PATH=$PATH:/usr/local/cuda/bin
