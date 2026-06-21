import pytest

from tensor_factory import review


@pytest.mark.unit
def test_only_approved_is_trainable():
    assert review.is_trainable(review.APPROVED)
    assert not review.is_trainable(review.PENDING)
    assert not review.is_trainable(review.REJECTED)


@pytest.mark.unit
def test_missing_or_unknown_review_is_untrusted():
    # The whole safety property: an unmarked label never trains.
    assert not review.is_trainable(None)
    assert not review.is_trainable("garbage")
    assert review.normalize(None) == review.PENDING
    assert review.normalize("garbage") == review.PENDING
    assert review.normalize(review.APPROVED) == review.APPROVED


@pytest.mark.unit
def test_review_summary_counts_and_trainable():
    coco = {
        "images": [
            {"id": 1, "review": review.APPROVED},
            {"id": 2, "review": review.PENDING},
            {"id": 3},  # missing -> pending
        ],
        "annotations": [
            {"id": 1, "review": review.APPROVED},
            {"id": 2, "review": review.APPROVED},
            {"id": 3, "review": review.PENDING},
            {"id": 4, "review": review.REJECTED},
            {"id": 5},  # missing -> pending
        ],
    }
    s = review.review_summary(coco)
    assert s["images"] == {"total": 3, "pending": 2, "approved": 1, "rejected": 0}
    assert s["annotations"]["total"] == 5
    assert s["annotations"]["trainable"] == 2
    assert s["annotations"]["pending"] == 2
    assert s["annotations"]["rejected"] == 1


@pytest.mark.unit
def test_review_summary_empty():
    s = review.review_summary({})
    assert s["images"]["total"] == 0
    assert s["annotations"]["trainable"] == 0
