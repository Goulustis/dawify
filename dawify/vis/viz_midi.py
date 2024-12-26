from dawify.mis_utils import rprint

import base64
import webbrowser
import os.path as osp
import glob

def to_data_url(midi_filename):
    """ This is crucial for Colab/WandB support. Thanks to Scott Hawley!!
        https://github.com/drscotthawley/midi-player/blob/main/midi_player/midi_player.py

    """
    with open(midi_filename, "rb") as f:
        encoded_string = base64.b64encode(f.read())
    return 'data:audio/midi;base64,'+encoded_string.decode('utf-8')

def create_html_from_midi(midifile, fname="PLACE HOLDER"):
    html_template = f"""
<!DOCTYPE html>
<html>
<head>
  <title>Awesome MIDI Player</title>
  <script src="https://cdn.jsdelivr.net/combine/npm/tone@14.7.58,npm/@magenta/music@1.23.1/es6/core.js,npm/focus-visible@5,npm/html-midi-player@1.5.0">
  </script>
  <style>
    /* Background color for the section */
    #proll {{background-color:transparent}}
    /* Custom player style */
    #proll midi-player {{
      display: block;
      width: inherit;
      margin: 4px;
      margin-bottom: 0;
      transform-origin: top;
      transform: scaleY(0.8); /* Added scaleY */
    }}
    #proll midi-player::part(control-panel) {{
      background: #d8dae880;
      border-radius: 8px 8px 0 0;
      border: 1px solid #A0A0A0;
    }}
    /* Custom visualizer style */
    #proll midi-visualizer .piano-roll-visualizer {{
      background: #45507328;
      border-radius: 0 0 8px 8px;
      border: 1px solid #A0A0A0;
      margin: 4px;
      margin-top: 1;
      overflow: auto;
      transform-origin: top;
      transform: scaleY(0.8); /* Added scaleY */
    }}
    #proll midi-visualizer svg rect.note {{
      opacity: 0.6;
      stroke-width: 2;
    }}
    #proll midi-visualizer svg rect.note[data-instrument="0"] {{
      fill: #e22;
      stroke: #055;
    }}
    #proll midi-visualizer svg rect.note[data-instrument="2"] {{
      fill: #2ee;
      stroke: #055;
    }}
    #proll midi-visualizer svg rect.note[data-is-drum="true"] {{
      fill: #888;
      stroke: #888;
    }}
    #proll midi-visualizer svg rect.note.active {{
      opacity: 0.9;
      stroke: #34384F;
    }}
    /* Media queries for responsive scaling */
    @media (max-width: 700px) {{ #proll midi-visualizer .piano-roll-visualizer {{transform-origin: top; transform: scaleY(0.75);}} }}
    @media (max-width: 500px) {{ #proll midi-visualizer .piano-roll-visualizer {{transform-origin: top; transform: scaleY(0.7);}} }}
    @media (max-width: 400px) {{ #proll midi-visualizer .piano-roll-visualizer {{transform-origin: top; transform: scaleY(0.6);}} }}
    @media (max-width: 300px) {{ #proll midi-visualizer .piano-roll-visualizer {{transform-origin: top; transform: scaleY(0.5);}} }}
  </style>
</head>
<body>
  <div>
    <a target="_blank" style="font-size: 14px;">{fname}</a> <br>
  </div>
  <div>
    <section id="proll">
      <midi-player src="{midifile}" sound-font="https://storage.googleapis.com/magentadata/js/soundfonts/sgm_plus" visualizer="#proll midi-visualizer">
      </midi-player>
      <midi-visualizer src="{midifile}">
      </midi-visualizer>
    </section>
  </div>
</body>
</html>
"""
    html = f"""<div style="display: flex; justify-content: center; align-items: center;">
                  <iframe style="width: 100%; height: 500px; overflow:hidden" srcdoc='{html_template}'></iframe>
            </div>"""
    return html


def display_midifiles(midi_files):
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MIDI Display</title>
        <style>
            body {
                text-align: center;
                background-color: #f0f0f0;
            }
            .larger-arrow {
                font-size: 2em;
                letter-spacing: 4em;
                display: inline-block;
            }
            .tbl_video {
                margin-bottom: 40px;
            }
            .container {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
    """

    for midi_file in midi_files:
        data_url = to_data_url(midi_file)
        file_name = osp.basename(midi_file)
        html_content += create_html_from_midi(data_url, file_name)

    html_content += """
        </div>
    </body>
    </html>
    """

    output_file = "/tmp/midi_files.html"
    with open(output_file, "w") as f:
        f.write(html_content)

    rprint(f"[blue]Click here to open: {output_file}[/blue]", print_trace=False)

    return html_content

    


if __name__ == "__main__":
    fold = "/home/boss/projects/dawify/outputs/mt3/All the Way North"
    midi_files = sorted(glob.glob(osp.join(fold, "*.mid")))
    display_midifiles(midi_files)