yt-dlp \
  --skip-download \
  --write-subs \
  --write-auto-subs \
  --sub-langs "en.*" \
  --convert-subs srt \
  -o "video.%(ext)s" \
  "YOUTUBE_URL"


cat *.srt | claude -p \
  "Summarize this video transcript with timestamps, key claims, and action items."

yt-dlp \
  -f "bv*+ba/b" \
  --merge-output-format mp4 \
  -o "video.%(ext)s" \
  "YOUTUBE_URL"


mkdir frames

ffmpeg \
  -i video.mp4 \
  -vf "fps=1/15,scale=1280:-1" \
  -q:v 2 \
  frames/frame-%04d.jpg