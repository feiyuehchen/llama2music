from miditoolkit import MidiFile

from statistics import mean
import os

from miditoolkit.midi import parser as mid_parser  
from miditoolkit.midi import containers as ct

save_dir = "../dataset/ailab17k_data/midi_data"
def traverse_dir(root_dir,
                extension, # extension=('mp3', 'wav')
                is_sort = False, 
                is_pure = False
                ):
    file_list = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path
                file_list.append(pure_path)
    
    if is_sort:
        print("sort the file list")
        file_list.sort()
    print(f"total count of the file: {len(file_list)}")
    return file_list

midi_list = traverse_dir(save_dir, ('mid', 'midi'))

test_path = "../dataset/ailab17k_data/midi_data/test/ailab17k/3.mid"
midi_obj = MidiFile(test_path)
print(midi_obj)

tpb = midi_obj.ticks_per_beat
print(tpb)
beat_resol = tpb
tempo_list = [tempo.tempo for tempo in midi_obj.tempo_changes]
print(mean(tempo_list)*4)
# define interval: from 2nd to 8th bar
st = beat_resol * 4 * 2
ed = beat_resol * 4 * 32
print(st, ed)

tmp = midi_obj.instruments[0].notes[-1].end 
tb = tmp/tpb

tempo_list = [tempo.tempo for tempo in midi_obj.tempo_changes]

print(tb*60/mean(tempo_list))
sec_const = 60/mean(tempo_list)/tpb
slice_time = 30
cur_start_sec = 0
cur_end_sec = 0 + slice_time


t = 100*beat_resol*mean(tempo_list)/60
print(t)
print(midi_obj.max_tick)

midi_list = []
tempo_list = [tempo.tempo for tempo in midi_obj.tempo_changes]
while cur_start_sec>tb*60/mean(tempo_list):
    mido_obj = mid_parser.MidiFile()
    print(cur_start_sec*mean(tempo_list)/60)
    for inst in midi_obj.instruments:
        temp_notes = []
        print(len(inst.notes))
        for note in inst.notes:
            note_time = note.start*sec_const
            if note_time < cur_start_sec:
                continue
            elif note_time >= cur_start_sec and note_time < cur_end_sec:
                temp_notes.append(note)
            else:
                break
        
        print(len(temp_notes))
        track = ct.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
        track.notes = temp_notes
        mido_obj.instruments.append(track)

    cur_start_sec+=slice_time
    cur_end_sec+=slice_time

    print(mido_obj)
    print("=")
    print(mido_obj.instruments[0].notes)
    print("="*100)
    midi_list.append(mido_obj)
rank = int(os.environ["RANK"])
print(rank)
print(midi_list)