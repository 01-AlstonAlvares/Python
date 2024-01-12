from pytube import YouTube
import os

def main():
    video_url = input('Enter the YouTube video URL: ')

    if os.name == 'nt':
        path = os.getcwd() + '\\'
    else:
        path = os.getcwd() + '/'

    try:
        yt = YouTube(video_url)
        video_stream = yt.streams.filter(only_audio=True).first()
        video_stream.download(output_path=path)

        # Rename the downloaded file to .mp3
        name = yt.title.replace(" ", "_")
        location = path + name + '.mp4'
        renametomp3 = path + name + '.mp3'

        os.rename(location, renametomp3)

        print(f'Download and conversion completed: {renametomp3}')

    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    main()
