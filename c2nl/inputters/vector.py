# src: https://github.com/facebookresearch/DrQA/blob/master/drqa/reader/vector.py

import torch


def vectorize(ex, model):
    """Torchify a single example."""
    src_dict = model.src_dict
    tgt_dict = model.tgt_dict

    code, summary = ex['code'], ex['summary']

    code_char_rep = None
    summ_char_rep = None
    code_type_rep = None
    code_mask_rep = None

    # Index words
    code_word_rep = torch.LongTensor(code.vectorize(word_dict=src_dict))
    summ_word_rep = torch.LongTensor(summary.vectorize(word_dict=tgt_dict))

    # Index chars
    if model.args.use_src_char:
        code_char_rep = torch.LongTensor(code.vectorize(word_dict=src_dict, _type='char'))
    if model.args.use_tgt_char:
        summ_char_rep = torch.LongTensor(summary.vectorize(word_dict=tgt_dict, _type='char'))

    if model.args.use_code_type:
        assert len(code.type) == len(code.tokens)
        code_type_rep = torch.LongTensor(code.type)

    if code.mask:
        code_mask_rep = torch.LongTensor(code.mask)

    # target is only used to compute loss during training
    target = torch.LongTensor(summary.vectorize(tgt_dict))

    return {
        'id': code.id,
        'language': code.language,
        'code_word_rep': code_word_rep,
        'code_char_rep': code_char_rep,
        'code_type_rep': code_type_rep,
        'code_mask_rep': code_mask_rep,
        'summ_word_rep': summ_word_rep,
        'summ_char_rep': summ_char_rep,
        'target': target,
        'code': code.text,
        'code_tokens': code.tokens,
        'summ': summary.text,
        'summ_tokens': summary.tokens,
        'src_vocab': code.src_vocab,
        'use_src_word': model.args.use_src_word,
        'use_tgt_word': model.args.use_tgt_word,
        'use_src_char': model.args.use_src_char,
        'use_tgt_char': model.args.use_tgt_char,
        'use_code_type': model.args.use_code_type,
        'use_code_mask': code_mask_rep is not None,
        'stype': summary.type
    }


def batchify(batch):
    """Gather a batch of individual examples into one batch."""

    # batch is a list of vectorized examples
    batch_size = len(batch)
    use_src_word = batch[0]['use_src_word']
    use_tgt_word = batch[0]['use_tgt_word']
    use_src_char = batch[0]['use_src_char']
    use_tgt_char = batch[0]['use_tgt_char']
    use_code_type = batch[0]['use_code_type']
    use_code_mask = batch[0]['use_code_mask']
    ids = [ex['id'] for ex in batch]
    language = [ex['language'] for ex in batch]

    # --------- Prepare Code tensors ---------
    code_words = [ex['code_word_rep'] for ex in batch]
    code_chars = [ex['code_char_rep'] for ex in batch]
    code_type = [ex['code_type_rep'] for ex in batch]
    code_mask = [ex['code_mask_rep'] for ex in batch]
    max_code_len = max([d.size(0) for d in code_words])

    # Batch Code Representations
    code_len_rep = torch.LongTensor(batch_size).zero_()
    code_word_rep = torch.LongTensor(batch_size,
                                     max_code_len).zero_() if use_src_word else None
    code_type_rep = torch.LongTensor(batch_size,
                                     max_code_len).zero_() if use_code_type else None
    code_mask_rep = torch.LongTensor(batch_size,
                                     max_code_len).zero_() if use_code_mask else None
    code_char_rep = torch.LongTensor(batch_size,
                                     max_code_len,
                                     code_chars[0].size(1)).zero_() if use_src_char else None

    for i in range(batch_size):
        code_len_rep[i] = code_words[i].size(0)
        if use_src_word:
            code_word_rep[i, :code_words[i].size(0)].copy_(code_words[i])
        if use_code_type:
            code_type_rep[i, :code_type[i].size(0)].copy_(code_type[i])
        if use_code_mask:
            code_mask_rep[i, :code_mask[i].size(0)].copy_(code_mask[i])
        if use_src_char:
            code_char_rep[i, :code_chars[i].size(0), :].copy_(code_chars[i])

    # --------- Prepare Summary tensors ---------
    summ_words = [ex['summ_word_rep'] for ex in batch]
    summ_chars = [ex['summ_char_rep'] for ex in batch]
    max_sum_len = max([q.size(0) for q in summ_words])

    # Batch Summaries
    summ_len_rep = torch.LongTensor(batch_size).zero_()
    summ_word_rep = torch.LongTensor(batch_size,
                                     max_sum_len).zero_() if use_tgt_word else None
    summ_char_rep = torch.LongTensor(batch_size,
                                     max_sum_len,
                                     summ_chars[0].size(1)).zero_() if use_tgt_char else None
    for i in range(batch_size):
        summ_len_rep[i] = summ_words[i].size(0)
        if use_tgt_word:
            summ_word_rep[i, :summ_words[i].size(0)].copy_(summ_words[i])
        if use_tgt_char:
            summ_char_rep[i, :summ_chars[i].size(0), :].copy_(summ_chars[i])

    # --------- Prepare other tensors ---------
    targets = [ex['target'] for ex in batch]
    max_tgt_length = max([t.size(0) for t in targets])
    tgt_tensor = torch.LongTensor(batch_size, max_tgt_length).zero_()
    for i, a in enumerate(targets):
        tgt_tensor[i, :a.size(0)].copy_(targets[i])

    # Prepare source vocabs, alignment [required for Copy Attention]
    source_maps = []
    alignments = []
    src_vocabs = []
    for j in range(batch_size):
        target = batch[j]['summ_tokens']
        context = batch[j]['code_tokens']
        vocab = batch[j]['src_vocab']
        src_vocabs.append(vocab)

        # Mapping source tokens to indices in the dynamic dict.
        src_map = torch.LongTensor([vocab[w] for w in context])
        source_maps.append(src_map)

        mask = torch.LongTensor([vocab[w] for w in target])
        alignments.append(mask)

    return {
        'ids': ids,
        'language': language,
        'batch_size': batch_size,
        'code_word_rep': code_word_rep,
        'code_char_rep': code_char_rep,
        'code_type_rep': code_type_rep,
        'code_mask_rep': code_mask_rep,
        'code_len': code_len_rep,
        'summ_word_rep': summ_word_rep,
        'summ_char_rep': summ_char_rep,
        'summ_len': summ_len_rep,
        'tgt_seq': tgt_tensor,
        'code_text': [ex['code'] for ex in batch],
        'code_tokens': [ex['code_tokens'] for ex in batch],
        'summ_text': [ex['summ'] for ex in batch],
        'summ_tokens': [ex['summ_tokens'] for ex in batch],
        'src_vocab': src_vocabs,
        'src_map': source_maps,
        'alignment': alignments,
        'stype': [ex['stype'] for ex in batch]
    }
