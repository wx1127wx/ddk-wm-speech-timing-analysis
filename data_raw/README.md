# Controlled-access input files (not distributed)

Place the following deidentified files in this directory. The public release intentionally contains this schema only.

## `ddk_trials.csv`

Required columns: `subject_id`, `trial_index`, `DDK rate (syll/s)`, `DDK regularity (ms)`, `DDK duration (ms)`, `pause regularity (ms)`, `pause duration (ms)`. Expected structure: 155 participants × 3 Single trials = 465 rows.

## `ddkwm_trials.csv`

Required columns: `subject_id`, `ddkwm_condition` (1 = Low, 2 = High), `trial_index` (1–10 within block), `wm_correct` (0/1), and the five acoustic columns listed above. Expected structure: 155 participants × 20 trials = 3,100 rows.

## `participants.csv`

Required columns: `subject_id`, `age`, `sex`, `edu`, `CMMS`. IDs must be pseudonymous. `edu` is years of education; CMMS is scored 0–30.

## `manual_automatic_paired.csv`

Long table with columns: `participant_id`, `condition` (`Single`, `Low`, `High`), `trial_index`, `outcome`, `manual`, `automatic`. `outcome` must be one of the five manuscript outcomes using the source labels `DDK rate (syll/s)`, `DDK regularity (ms)`, `DDK duration (ms)`, `pause regularity (ms)`, or `pause duration (ms)`. Expected structure: 20 participants × 23 recordings × 5 outcomes = 2,300 rows.

Do not include original participant numbers, recording dates, filenames, audio paths, or raw audio in a public copy.
