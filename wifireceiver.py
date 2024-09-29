import numpy as np
import sys
import commpy as comm
from wifitransmitter import WifiTransmitter
import commpy.channelcoding.convcode as check

def WifiReceiver(output, level):
    return 0,0,0

import numpy as np

def undo_interleave(output, Interleave, nfft):
    nsym = int(len(output) / (2 * nfft))
    bits = np.zeros_like(output)

    for i in range(nsym):
        symbol = output[i * 2 * nfft:(i + 1) * 2 * nfft]
        bits[i * 2 * nfft:(i + 1) * 2 * nfft] = symbol[np.argsort(Interleave - 1)]
    
    return bits

def test():
    message = "hello"
    level = 1
    nfft = 64
    Interleave = np.reshape(np.transpose(np.reshape(np.arange(1, 2 * nfft + 1, 1), [-1, 4])), [-1,])

    if level >= 1:
        bits = np.unpackbits(np.array([ord(c) for c in message], dtype=np.uint8))
        print("Original bits before padding:")
        print(bits)
        
        bits = np.pad(bits, (0, 2 * nfft - len(bits) % (2 * nfft)), 'constant')
        nsym = int(len(bits) / (2 * nfft))
        output = np.zeros(shape=(len(bits), ))

        for i in range(nsym):
            symbol = bits[i * 2 * nfft:(i + 1) * 2 * nfft]
            output[i * 2 * nfft:(i + 1) * 2 * nfft] = symbol[Interleave - 1]

    print("\nInterleaved bits:")
    print(output)

    original_bits = undo_interleave(output, Interleave, nfft)
    print("\nBits after undoing interleaving:")
    print(original_bits)

    trimmed_bits = original_bits[:len(message) * 8].astype(np.uint8)
    print("\nTrimmed bits (should match original bits):")
    print(trimmed_bits)

    byte_array = np.packbits(trimmed_bits).tobytes()
    recovered_message = ''.join([chr(b) for b in byte_array])
    print("\nRecovered message:")
    print(recovered_message)

test()




