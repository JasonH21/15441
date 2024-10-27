import numpy as np
import sys
import commpy as comm
from wifitransmitter import WifiTransmitter
import commpy.channelcoding.convcode as check

def undo_interleave(output, Interleave, nfft):
    nsym = int(len(output) / (2 * nfft))
    bits = np.zeros_like(output)
    #uses majority vote on groups of 3 bits
    for i in range(nsym):
        symbol = output[i * 2 * nfft:(i + 1) * 2 * nfft]
        bits[i * 2 * nfft:(i + 1) * 2 * nfft] = symbol[np.argsort(Interleave - 1)]
    return bits

#obtain the length and the message without the length
def undo_length_encoding(output, nfft):
    len_binary_encoded = output[:2 * nfft]
    length_binary = []
    for i in range(0, len(len_binary_encoded), 3):
        group = len_binary_encoded[i:i + 3]
        majority_bit = int(np.round(np.mean(group)))
        length_binary.append(majority_bit)
    len_dec = int(''.join(map(str, length_binary)), 2)
    out_wo_len = output[2 * nfft:]
    return len_dec, out_wo_len


#implement the viterbi algorithm to decode convolutionally encoded bits
def viterbi_dec_custom(received_bits, trellis):
    n_states = trellis.number_states
    in_sym = trellis.number_inputs
    n = trellis.n 

    path_mets = np.full(n_states, np.inf)
    path_mets[0] = 0  
    paths = {state: [] for state in range(n_states)}

    next_state_table = trellis.next_state_table
    output_table = trellis.output_table

    n_steps = len(received_bits) // n
    for step in range(n_steps):
        rec_sym = received_bits[step * n:(step + 1) * n]
        temp_path_mets = np.full(n_states, np.inf)
        new_paths = {state: [] for state in range(n_states)}
        #iterate through the states
        for curr_state in range(n_states):
            if path_mets[curr_state] < np.inf:
                #iterate through the symbols
                for input_symbol in range(in_sym):
                    next_state = next_state_table[curr_state][input_symbol]
                    ex_output = output_table[curr_state][input_symbol]
                    ex_output_bits = np.array(
                        [int(x) for x in np.binary_repr(ex_output, width=n)]
                    )
                    #calculate hamming distance
                    hamm_dist = np.sum(rec_sym != ex_output_bits)
                    metric = path_mets[curr_state] + hamm_dist
                    if metric < temp_path_mets[next_state]:
                        temp_path_mets[next_state] = metric
                        new_paths[next_state] = paths[curr_state] + [input_symbol]
        path_mets = temp_path_mets
        paths = new_paths

    #select the best state
    best_state = np.argmin(path_mets)
    decoded_bits = paths[best_state]

    return np.array(decoded_bits)

#module to handle the receiving of the message, implementation changes based on level
def WifiReceiver(output, level):
    nfft = 64
    preamble_len = 128 
    bpsym = int(np.log2(4))  
    pre_sym = preamble_len // bpsym  
    pre_chks = int(np.ceil(pre_sym / nfft))
    Interleave = np.reshape(
        np.transpose(
            np.reshape(np.arange(1, 2 * nfft + 1), [-1, 4])
        ), [-1]
    )
    cc1 = check.Trellis(np.array([3]), np.array([[0o7, 0o5]]))
    mod = comm.modulation.QAMModem(4)  
    #level 1
    if level == 1:
        begin_zero_padding = 0  
        len_dec, out_wo_len = undo_length_encoding(output, nfft)
        orig_bits = undo_interleave(out_wo_len, Interleave, nfft)
        trimmed_bits = orig_bits[:len_dec * 8].astype(np.uint8)
        byte_array = np.packbits(trimmed_bits).tobytes()
        rec_mess = ''.join([chr(b) for b in byte_array])
        return begin_zero_padding, rec_mess, len_dec
    #level 2
    if level == 2:
        begin_zero_padding = 0 
        mod = comm.modulation.QAMModem(4)
        demod_out = mod.demodulate(output, demod_type='hard')
        output_without_preamble = demod_out[preamble_len:]
        len_dec, out_wo_len = undo_length_encoding(
            output_without_preamble, nfft
        )
        dec_mess = viterbi_dec_custom(out_wo_len.astype(int),cc1)
        orig_bits = undo_interleave(dec_mess.astype(int), Interleave, nfft)
        trimmed_bits = orig_bits[:len_dec * 8].astype(np.uint8)
        byte_array = np.packbits(trimmed_bits).tobytes()
        rec_mess = ''.join([chr(b) for b in byte_array])
        return begin_zero_padding, rec_mess, len_dec
    #level 3
    if level == 3:
        begin_zero_padding = 0  
        bpsym = int(np.log2(4)) 
        pre_sym = preamble_len // bpsym  
        pre_chks = pre_sym // nfft  

        nsym = int(len(output) / nfft)
        symbols = np.zeros(nsym * nfft, dtype=complex)
        for i in range(nsym):
            td_sym = output[i * nfft:(i + 1) * nfft]
            fd_sym = np.fft.fft(td_sym)
            symbols[i * nfft:(i + 1) * nfft] = fd_sym

        sym_wo_pre = symbols[pre_chks * nfft:]

        mod = comm.modulation.QAMModem(4)
        demod_out = mod.demodulate(sym_wo_pre, demod_type='hard')

        len_dec, out_wo_len = undo_length_encoding(
            demod_out, nfft
        )
        dec_mess = viterbi_dec_custom(out_wo_len.astype(int),cc1)
        orig_bits = undo_interleave(dec_mess.astype(int), Interleave, nfft)
        trim_bits = orig_bits[:len_dec * 8].astype(np.uint8)
        byte_array = np.packbits(trim_bits).tobytes()
        rec_mess = ''.join([chr(b) for b in byte_array])
        return begin_zero_padding, rec_mess, len_dec
    #level 4
    if level == 4:
        #preamble stolen from the Transmitter
        preamble_bits = np.array([
            1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,
            1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1,
            1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0,
            1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1,
            1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,
            1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1,
            1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0,
            1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1
        ])

        bpsym = int(np.log2(4))  
        pre_sym = len(preamble_bits) // bpsym  

        pre_mod = mod.modulate(preamble_bits.astype(bool))

        num_pre_chks = int(np.ceil(len(pre_mod) / nfft))
        pre_td = np.array([])
        #oterate through chunks
        for i in range(num_pre_chks):
            fd_sym = pre_mod[i * nfft:(i + 1) * nfft]
            if len(fd_sym) < nfft:
                fd_sym = np.pad(fd_sym, (0, nfft - len(fd_sym)), 'constant')
            td_sym = np.fft.ifft(fd_sym)
            pre_td = np.concatenate((pre_td, td_sym))
        #make use of built in numpy functions to calculate correlation
        correlation = np.abs(np.correlate(output, pre_td, mode='valid'))
        start_index = np.argmax(correlation)
        begin_zero_padding = start_index  

        rec_sig = output[start_index:]

        num_full_symbols = len(rec_sig) // nfft
        rec_sig = rec_sig[:num_full_symbols * nfft]

        nsym = int(len(rec_sig) / nfft)
        symbols = np.zeros(nsym * nfft, dtype=complex)
        for i in range(nsym):
            td_sym = rec_sig[i * nfft:(i + 1) * nfft]
            fd_sym = np.fft.fft(td_sym)
            symbols[i * nfft:(i + 1) * nfft] = fd_sym

        slicing_index = num_pre_chks * nfft
        total_symbols = nsym * nfft
        if slicing_index > total_symbols:
            print("Error: Preamble length exceeds total number of symbols.")
            return None
        sym_wo_pre = symbols[slicing_index:]
        demod_out = mod.demodulate(sym_wo_pre, demod_type='hard')
        len_dec, out_wo_len = undo_length_encoding(
            demod_out, nfft
        )
        dec_mess = viterbi_dec_custom(out_wo_len.astype(int),cc1)
        orig_bits = undo_interleave(dec_mess.astype(int), Interleave, nfft)
        trimmed_bits = orig_bits[:len_dec * 8].astype(np.uint8)
        try:
            byte_array = np.packbits(trimmed_bits).tobytes()
            rec_mess = ''.join([chr(b) for b in byte_array])
        except ValueError:
            rec_mess = ""
        return begin_zero_padding, rec_mess, len_dec

#tester
for level in [1,2,3,4]:
    if level == 4:
        noise_pad_begin, txsignal, length = WifiTransmitter("hello world", level)
    else:
        txsignal = WifiTransmitter("hello world", level)
    print(WifiReceiver(txsignal, level))