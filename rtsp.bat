"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe" -rtsp_transport tcp -timeout 5000000 -i "rtsp://admin:123456@188.162.55.251:1601/stream2" -loglevel verbose -f rawvideo -pix_fmt bgr24 -vsync 2 -reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 2 -