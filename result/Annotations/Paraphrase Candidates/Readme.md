## Files of Interest

The "paraphrase candidate" files include the annotation as done by the lead author -- these mainly include a yes/no label for is a paraphrase candidate. It also includes a label for "repetition" paraphrases and with less consistency other labels like "high lexical similarity" etc. This directory also includes the samples for annotation ("Paraphrase Candidate Samples for Annotation").

### Paraphrase Candidate Samples for Annotation

**100 pilot**: 

`Sampled_100_PC_IDs.tsv` includes the sampled paraphrase candidates for the pilot 100 item annotation. Statistics as printed in generation:


			PARAPHRASES: 50 TOTAL
				Unclear paraphrases: 10
				Clear paraphrases: 40
				Repetition Paraphrases: 20
				Universal Paraphrases: 15
				Perspective Shift Paraphrases: 8
				High Similarity Paraphrases: 30
				Situational Paraphrases: 9
				Directional Paraphrases: 11
	
			NON-PARAPHRASES: 50 Total
				Missing context: 10
				High Lexical Similarity: 10
				Partial: 10
				(Clearly) unrelated: 13
				Related: 20
				CONCLUSION: 11

**500 further annotations**:

`Annotation-Sample-500.tsv` (see `23-09-11_23-09-15.txt`). Statistics as printed in generation:

							# Errors: 11
							# Context Missing: 133
							# Ambigous: 23
							# Paraphrases: 511
								 # Repetitions: 200
								 # Non-Repetitions: 311
							# Non-Paraphrases: 3022
							# Difficult (Non-Related) Non-Paraphrases: 22
							# Related (incl. difficult) Non-Paraphrases: 231
							Out of a sample pool of 3556, sampled 100 unique IDs, with
								 Sampled 2 ambiguous, initial ratio was 0.0064679415073115865
								 Sampled 10 repetitions, initial ratio was 0.0562429696287964
								 Sampled 6 non-repetition paraphrases, initial ratio was 0.0874578177727784
								 Sampled 82 non-paraphrases, initial ratio was 0.8498312710911136
									 Sampled 2 difficult non-paraphrases, initial ratio was 0.006186726659167604
									 Sampled 10 topically related non-paraphrases, initial ratio was 0.06496062992125984
							Further sampled 400 unique IDs
								 Sampled 40 repetitions, now totaling 50
								 Sampled 294 non-repetition paraphrases, now totaling 300
								 Sampled 66 non-paraphrases, now totaling 148
									 Sampled 16 difficult non-paraphrases, now totaling 18
									 Sampled 40 topically related non-paraphrases, now totaling 50		



### Paraphrase Candidate Annotation files

**Files on which 100 pilot annotations were sampled**:

```
ANON_23-06-20_post-sampling-updates_359-908_updated-PCs.tsv
ANON_23-06-20_post-sampling-updates_959-1158_annotations.tsv
```

these include only yes/no paraphrase annotations and some labels.

**Files on which 500 annotations were sampled:**

```
Qualtrics/
ANON_23-09_checked-ambig.tsv  # overwrites some annotations from Qualtrics with "AMBIGUOUS" label
ANON_23-09_checked-context.tsv  # overwrites some annotations from Qualtrics with "CONTEXT" label
```

some of them only include binary annotations of paraphrase yes/no while the rest in the "Qualtrics" folder also include annotator rationale answers.

```
["ANON_23-06-23_Paraphrase-Candidate_1159-1208_July+25,+2023_03.47.tsv",
"ANON_23-07-10_Paraphrase-Candidate_1209-1258_July+25,+2023_03.47.tsv",
"ANON_23-07-17_Paraphrase-Candidate_1259-1308_July+25,+2023_03.48.tsv",
"ANON_23-07-17_Paraphrase-Candidate_1309-1358_July+25,+2023_03.48.tsv",
"ANON_23-07-17_Paraphrase-Candidate_1359-1408_July+25,+2023_03.48.tsv"]
```



## Legacy Files

Unless you want to reproduce our sampling you need not worry about the legacy files. In `legacy files` you will find annotations of items by the lead author that you probably won't need to study. The reason for these is that to make annotations consistent some instances had to be re-labelled (i.e., `359-908_updated-PCs.tsv` ). Sampling for the 100 pilot study was done on `959-1158_annotations.tsv` and `359-908_updated-PCs.tsv`.  Some instances were again re-labeled after the sampling of 100 items (no main label shift of repetition/no paraphrase/paraphrase) but mainly fixing inconsistencies in the other used labels (like ONLY-RELATED see below). 

**In more detail:**

`959-1158_annotations.tsv` are annotations with the current annotations scheme on the last 959 - 1158. Note: there is no good reason 909-958 is missing. I made an error in generating the annotation surveys.

`359-908_first-pass.tsv` includes annotations with the options "is clearly not paraphrasing anything the guest said" and "is or is possibly paraphrasing something the guest said". Moreover in the comments field, I added the following keywords consistently for all examples: CONTEXT (i.e., context information is missing), CONCLUSION, REPETITION. Some other keywords (REFERENCE, IT-REFERENCE, INSPIRED-REFERENCE, PRAGMATIC) are also added but I did NOT necessarily keep annotating them for the whole set (!) . `359-908_updated-PCs.tsv` includes the original annotations from `359-908_first-pass.tsv`. Moreover in the comments field of the examples  where "is or is possibly paraphrasing something the guest said" is selected and it's not marked as a repetition (remaining set of 115 pairs), I added consistent annotations of the following keywords: HIGH-LEXICAL-SIMILARITY, CLEAR-PARAPHRASE (not necessarily easy), CLEAR-NON-PARAPHRASE, UNIVERSAL, AMBIGUOUS. Overview of other keywords:

	CATEGORY 1: properties independent from paraphrase or not
					
	    PRAGMATIC (sth makes it non-universal, better: context additional attribute)
		    BACKGROUND-KNOWLEDGE (add in background knowledge)
		    COREFERENCE (e.g., they refers to ENTITY before)
		    PERSPECTIVE-SHIFT (mostly i/you)
	
	    HIGH-LEXICAL-SIMILARITY ("high" lexical word overlap)
	        MOSTLY-REPETITION (or: REPETITION, is a repetition or almost repetition present)
	
	    DIFFICULT (can without 'ambiguous' still be 'clear')
	        MISREPRESENTATION
	        NEGATED-PARAPHRASE (negate instead of rephrase directly)
	        SURROGATE-PARAPHRASE (rephrase through saying someone else sees it like this as well: ...)
			CONTEXT (missing)
	
			DIRECTIONAL (e.g., summary or conclusion-like, could be understood as "paraphrase" in one direction but not the other)
				META-COMMENT (commenting on the interviewer/behavior instead of repeating a poitn)
				CONCLUSION (interpretation, conclusion)
						
	CATEGORY 2: related to whether it's a paraphrase or not
	
	    CLEAR-PARAPHRASE
	        UNIVERSAL (ignore coreference troubles)
	        SIDE-PARAPHRASE (i.e., mentioned in passing, more "boring")
	    
	    CLEAR-NON-PARAPHRASE
	        ONLY-RELATED (before: INSPIRED-REFERENCE)
	            PARTIAL (a subselection could be understood as a paraphrase, but not the whole)
			UNRELATED
	
	    AMBIGUOUS

Attention: some labels were update AFTER sampling (especially those that had incomplete annotations for the repetition cases after the first pass) --> this leads to different statistics than was sampled for initially. `23-06-20_post-sampling-updates_359-908_updated-PCs.tsv` and `23-06-20_post-sampling-updates_959-1158_annotations.tsv` again updated but AFTER sampling the 100 PCs, to be reproducible in the sampling `359-908_updated-PCs.tsv` and `959-1158_annotations.tsv` needs to be used.

