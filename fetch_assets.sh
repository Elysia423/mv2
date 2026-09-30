#!/usr/bin/env bash
# Download fonts (Google Fonts, OFL) and the brain MRI sample used by the MRI vignette.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p assets/fonts
B=https://raw.githubusercontent.com/google/fonts/main/ofl
get() { [ -s "assets/fonts/$2" ] || curl -sSfL -o "assets/fonts/$2" "$B/$1"; }
get "notoserifsc/NotoSerifSC%5Bwght%5D.ttf" NotoSerifSC.ttf
get "notosanssc/NotoSansSC%5Bwght%5D.ttf" NotoSansSC.ttf
get "mashanzheng/MaShanZheng-Regular.ttf" MaShanZheng.ttf
get "montserrat/Montserrat%5Bwght%5D.ttf" Montserrat.ttf
get "notomusic/NotoMusic-Regular.ttf" NotoMusic.ttf
get "jetbrainsmono/JetBrainsMono%5Bwght%5D.ttf" JetBrainsMono.ttf
get "cormorantgaramond/CormorantGaramond%5Bwght%5D.ttf" Cormorant.ttf
get "notoemoji/NotoEmoji%5Bwght%5D.ttf" NotoEmoji.ttf
[ -s assets/brain.npy ] || python3 -c "import numpy as np, skimage.data as d; np.save('assets/brain.npy', d.brain())"
echo "assets ready"
