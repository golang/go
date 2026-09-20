use std::{cell::RefCell, fs, process};

use libafl::{
    corpus::{inmemory::TestcaseStorage, CachedOnDiskCorpus, Corpus, CorpusId, Testcase},
    inputs::BytesInput,
};

#[test]
fn cached_on_disk_add_sets_testcase_id() {
    let corpus_dir =
        std::env::temp_dir().join(format!("gosentry-cached-corpus-id-test-{}", process::id()));
    let _ = fs::remove_dir_all(&corpus_dir);

    let mut corpus = CachedOnDiskCorpus::new(&corpus_dir, 1).unwrap();
    let id = corpus
        .add(Testcase::new(BytesInput::new(vec![0x41])))
        .unwrap();

    {
        let mut testcase = corpus.get(id).unwrap().borrow_mut();
        corpus.load_input_into(&mut testcase).unwrap();
        assert_eq!(testcase.corpus_id(), Some(id));
    }

    drop(corpus);
    fs::remove_dir_all(corpus_dir).unwrap();
}

#[test]
fn explicit_id_insertion_sets_testcase_id() {
    let mut storage = TestcaseStorage::new();
    let id = CorpusId::from(0_usize);

    storage
        .insert_inner_with_id(
            RefCell::new(Testcase::new(BytesInput::new(vec![0x42]))),
            false,
            id,
        )
        .unwrap();

    assert_eq!(
        storage.enabled.get(id).unwrap().borrow().corpus_id(),
        Some(id)
    );
}
