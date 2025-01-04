from enum import IntEnum


class AminoAcids(IntEnum):
    """IntEnum representing the amino acids."""

    ALANINE = 1
    ARGININE = 2
    ASPARAGINE = 3
    ASPARTIC_ACID = 4
    CYSTEINE = 5
    GLUTAMIC_ACID = 6
    GLUTAMINE = 7
    GLYCINE = 8
    HISTIDINE = 9
    ISOLEUCINE = 10
    LEUCINE = 11
    LYSINE = 12
    METHIONINE = 13
    PHENYLALANINE = 14
    PROLINE = 15
    SERINE = 16
    THREONINE = 17
    TRYPTOPHAN = 18
    TYROSINE = 19
    VALINE = 20


AMINO_ACIDS_TO_CHARGES_DICT = {
    AminoAcids.ALANINE.value: 0,  # Neutral
    AminoAcids.ARGININE.value: +1,  # Positive
    AminoAcids.ASPARAGINE.value: 0,  # Neutral
    AminoAcids.ASPARTIC_ACID.value: -1,  # Negative
    AminoAcids.CYSTEINE.value: 0,  # Neutral
    AminoAcids.GLUTAMIC_ACID.value: -1,  # Negative
    AminoAcids.GLUTAMINE.value: 0,  # Neutral
    AminoAcids.GLYCINE.value: 0,  # Neutral
    AminoAcids.HISTIDINE.value: +1,  # Positive (weakly at pH 7)
    AminoAcids.ISOLEUCINE.value: 0,  # Neutral
    AminoAcids.LEUCINE.value: 0,  # Neutral
    AminoAcids.LYSINE.value: +1,  # Positive
    AminoAcids.METHIONINE.value: 0,  # Neutral
    AminoAcids.PHENYLALANINE.value: 0,  # Neutral
    AminoAcids.PROLINE.value: 0,  # Neutral
    AminoAcids.SERINE.value: 0,  # Neutral
    AminoAcids.THREONINE.value: 0,  # Neutral
    AminoAcids.TRYPTOPHAN.value: 0,  # Neutral
    AminoAcids.TYROSINE.value: 0,  # Neutral
    AminoAcids.VALINE.value: 0,  # Neutral
}
