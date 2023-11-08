import torch
import numpy as np
from typing import Iterable
from functools import partial
from collections import defaultdict
from typing import List

def minimun_edit_distance(reference:str,candidate:str,insert_cost:float=1.,delete_cost:float=1.,sub_cost:float=2.):
    ref_len = len(reference)
    can_len = len(candidate)

    dynamic_table = np.empty(shape=(can_len+1,ref_len+1))
    dynamic_table[0,:] = np.arange(ref_len+1)*insert_cost
    dynamic_table[:,0] = np.arange(can_len+1)*delete_cost
    for r in range(1,can_len+1):
        for c in range(1,ref_len+1):
            if candidate[r-1] != reference[c-1]:
                sub_edit = dynamic_table[r-1,c-1] + sub_cost
            else:
                sub_edit = dynamic_table[r - 1, c - 1]
            min_edit = min(dynamic_table[r,c-1]+insert_cost,dynamic_table[r-1,c]+delete_cost,sub_edit)
            dynamic_table[r,c] = min_edit
    return dynamic_table[-1,-1]

def thresold_candidates(reference:str,candidates:Iterable[str],thresold:int=1):
    partial_func = partial(minimun_edit_distance,reference=reference)
    thresold_candidates = []
    for can in candidates:
        if partial_func(candidate=can) <= thresold:
            thresold_candidates.append({can:partial_func(candidate=can)})
    d = dict({
        thresold : tuple(thresold_candidates)
    })
    return d

def minimun_edit_distance_thresold(references:torch.Tensor,candidates:torch.Tensor,thresold:int=4):
    if isinstance(references,torch.Tensor) and isinstance(candidates,torch.Tensor):
        assert len(references.shape) >= 2, f"'References expected len shape 2 ([N,setence_len]), but got len of {len((references.shape))}'"
        assert len(candidates.shape) >= 2, f"'Candidates expected len shape 2 ([N,setence_len]), but got len of {len((references.shape))}'"
        refs_as_list = map(lambda l: ''.join(map(str,l)),references.to(torch.int8).numpy().tolist())
        cans_as_list = map(lambda l: ''.join(map(str,l)),candidates.to(torch.int8).numpy().tolist())
    else:
        refs_as_list = map(lambda l: ''.join(map(str,l)),references)
        cans_as_list = map(lambda l: ''.join(map(str,l)), candidates)
    d = dict()
    for ref in refs_as_list:
        d[ref] = thresold_candidates(reference=ref,candidates=cans_as_list,thresold=thresold)
    return d

def ctc_filter(alignment_iter:torch.Tensor|Iterable[int|str],blank:int|str=0) -> List[List[int]]:
    alignment_without_blank = []
    alignment_without_blank.append([])
    for c in alignment_iter.numpy():
        if c != blank:
            alignment_without_blank[-1].append(c)
        else:
            alignment_without_blank.append([])
    alignment_without_blank = filter(lambda x: x,alignment_without_blank)
    def merge_alignment(alignment):
        merged = [alignment[0]]
        for c in alignment[1:]:
            if c != merged[-1]:
                merged.append(c)
        return merged
    merged_split = list(map(merge_alignment,alignment_without_blank))
    return merged_split

def reverse_target(target:str,alignments:Iterable[Iterable[int|str]],blank:str|int=0):
    d = dict()
    mappable_algnments = list()
    for alignment in alignments:
        if ctc_filter(alignment_iter=alignment,blank=blank) == target:
            mappable_algnments.append(alignment)
    d[target] = mappable_algnments
    return d

def ctc_inference(softmax_output:torch.Tensor,beam_width:int=1,blank_index:int=0):
    T,C = softmax_output.shape
    empty = tuple()
    beam = [(empty,(1,0))]

    for t in range(T):
        next_beam = defaultdict(lambda : (0,0))
        for c in range(C):
            p = softmax_output[t,c]

            for prefix, (p_b,p_nb) in beam:
                if c == blank_index:
                    n_p_b, n_p_nb = next_beam[prefix]
                    n_p_b += (p_b + p_nb) * p
                    next_beam[prefix] = n_p_b,n_p_nb
                else:
                    last_t = prefix[-1] if prefix else None
                    n_prefix = prefix + (c,)
                    n_p_b, n_p_nb = next_beam[n_prefix]
                    if c != last_t:
                        n_p_nb += (p_b + p_nb) * p
                    else:
                        n_p_nb += p_b * p
                    next_beam[n_prefix] = n_p_b,n_p_nb

                    if c == last_t:
                        n_p_b, n_p_nb = next_beam[prefix]
                        n_p_nb += p_nb * p
                        next_beam[prefix] = (n_p_b, n_p_nb)

        beam = sorted(next_beam.items(),key=lambda x: sum(x[1]),reverse=True)[:beam_width]
    best_beam = beam[0]
    return best_beam

if __name__ == '__main__':
    pass






