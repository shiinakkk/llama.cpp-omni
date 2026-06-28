import struct,wave,math,os
d='tools/omni/output/tts_wav/'
files=sorted([f for f in os.listdir(d) if f.startswith('wav_') and f.endswith('.wav')])
if not files: print("NO WAV FILES"); exit(1)
for f in files:
    p=os.path.join(d,f)
    with wave.open(p,'r') as w:
        n=w.getnframes(); data=w.readframes(n)
        s=struct.unpack(f'{n}h',data)
        rms=int(math.sqrt(sum(x*x for x in s)/n)) if n>0 else 0
        peak=max(abs(x) for x in s) if n>0 else 0
        nz=sum(1 for x in s if x!=0)
        print(f'{f}: rms={rms} peak={peak} nz={nz}/{n}')
