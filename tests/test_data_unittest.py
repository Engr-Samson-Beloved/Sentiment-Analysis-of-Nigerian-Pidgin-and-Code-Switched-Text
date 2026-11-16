"""Data-only regression checks runnable without sklearn or pytest."""
import unittest
from pathlib import Path
import pandas as pd
from naija_sentiment.data.ingestion import read_dataset
from naija_sentiment.data.validation import validate, statistics
from naija_sentiment.data.cleaning import clean_text
from naija_sentiment.data.deduplication import duplicate_groups
from naija_sentiment.data.annotation import validate_annotations, adjudicate
from naija_sentiment.data.agreement import fleiss_kappa
from naija_sentiment.data.permissions import audit_permissions, read_permissions, validate_permission_ledger
from naija_sentiment.data.privacy import privacy_audit
from naija_sentiment.data.splitting import split_dataset
from naija_sentiment.data.versioning import dataframe_sha256, split_manifest
from naija_sentiment.evaluation.metrics import metrics
from naija_sentiment.evaluation.reporting import experiment_record, validate_experiment_record
from naija_sentiment.evaluation.error_analysis import error_summary, error_table
from naija_sentiment.evaluation.bias_analysis import language_gap
from naija_sentiment.evaluation.learning_curves import training_sizes

ROOT = Path(__file__).resolve().parents[1]
class DataChecks(unittest.TestCase):
    def setUp(self):
        self.df = read_dataset(ROOT / 'data/samples/synthetic.csv', True)
        self.a = pd.read_csv(ROOT / 'data/samples/synthetic_annotations.csv', dtype=str, keep_default_na=False)
    def test_csv_and_statistics(self):
        self.assertEqual(len(self.df), 90)
        self.assertEqual(statistics(self.df)['sentiment'], {'positive':30, 'negative':30, 'neutral':30})
    def test_missing_column(self):
        with self.assertRaises(ValueError): validate(self.df.drop(columns='text'))
    def test_invalid_label(self):
        self.df.loc[0, 'sentiment'] = 'unknown'
        with self.assertRaises(ValueError): validate(self.df)
    def test_missing_text(self):
        self.df.loc[0, 'text'] = '  '
        with self.assertRaises(ValueError): validate(self.df)
    def test_duplicate_ids(self):
        self.df.loc[1, 'text_id'] = self.df.loc[0, 'text_id']
        with self.assertRaises(ValueError): validate(self.df)
    def test_unicode_and_redaction(self):
        self.assertEqual(clean_text(' Ẹ káàárọ̀! 😃  ɗ '), 'Ẹ káàárọ̀! 😃 ɗ')
        self.assertEqual(clean_text('@person a@b.com https://example.org +234 801 234 5678'), '[USER] [EMAIL] [URL] [PHONE]')
    def test_exact_and_near_duplicates(self):
        tiny = pd.DataFrame({'text':['this is a very pleasant day!', 'this is a very pleasant day!', 'this is a very pleasant day?', 'completely unrelated']})
        groups = duplicate_groups(tiny)
        self.assertEqual(groups[0], groups[1]); self.assertEqual(groups[0], groups[2]); self.assertNotEqual(groups[0], groups[3])
    def test_related_group(self):
        tiny = pd.DataFrame({'text':['happy', 'awful', 'ordinary'], 'group_id':['thread1','thread1','']})
        groups = duplicate_groups(tiny)
        self.assertEqual(groups[0], groups[1]); self.assertNotEqual(groups[0], groups[2])
    def test_three_annotators(self):
        validate_annotations(self.a, self.df.text_id)
        with self.assertRaises(ValueError): validate_annotations(self.a.iloc[1:], self.df.text_id)
    def test_repeated_annotator(self):
        self.a.loc[1, 'annotator_id'] = self.a.loc[0, 'annotator_id']
        with self.assertRaises(ValueError): validate_annotations(self.a, self.df.text_id)
    def test_kappa_unanimous(self):
        self.assertEqual(fleiss_kappa(self.a)['fleiss_kappa'], 1)
    def test_kappa_disagreement(self):
        a = pd.DataFrame({'text_id':['a']*3 + ['b']*3, 'sentiment_label':['positive','negative','neutral']*2})
        self.assertAlmostEqual(fleiss_kappa(a)['fleiss_kappa'], -.5)
    def test_kappa_undefined(self):
        self.a['sentiment_label'] = 'positive'
        self.assertIsNone(fleiss_kappa(self.a)['fleiss_kappa'])
    def test_consensus_and_tie(self):
        consensus, queue = adjudicate(self.a)
        self.assertEqual(len(consensus), 90); self.assertTrue(queue.empty)
        self.a.loc[:2, 'sentiment_label'] = ['positive', 'negative', 'neutral']
        consensus, queue = adjudicate(self.a)
        self.assertEqual(consensus.set_index('text_id').loc['synthetic-0','sentiment'], '')
        self.assertEqual(len(queue), 3)
    def test_majority(self):
        self.a.loc[:2, 'sentiment_label'] = ['negative','negative','positive']
        consensus, queue = adjudicate(self.a)
        self.assertEqual(consensus.set_index('text_id').loc['synthetic-0','sentiment'], 'negative')
        self.assertEqual(len(queue), 3)
    def test_permission_audit(self):
        ledger = read_permissions(ROOT / 'data/samples/synthetic_permissions.csv')
        report = audit_permissions(self.df, ledger)
        self.assertTrue(report['ready_for_research_use'])
        ledger.loc[0, 'permission_status'] = 'pending'
        self.assertFalse(audit_permissions(self.df, ledger)['ready_for_research_use'])
    def test_permission_ledger_rejects_missing_evidence(self):
        ledger = read_permissions(ROOT / 'data/samples/synthetic_permissions.csv')
        ledger.loc[0, 'evidence_location'] = ''
        with self.assertRaises(ValueError): validate_permission_ledger(ledger)
    def test_privacy_audit(self):
        self.assertTrue(privacy_audit(self.df)['automated_screen_passed'])
        self.df.loc[0, 'text'] = 'Contact test@example.org or +234 801 234 5678'
        report = privacy_audit(self.df)
        self.assertFalse(report['automated_screen_passed'])
        self.assertEqual(report['finding_counts']['email'], 1)
        self.assertEqual(report['finding_counts']['phone'], 1)
    def test_split_is_reproducible_and_leakage_free(self):
        first=split_dataset(self.df); second=split_dataset(self.df)
        for name in first: pd.testing.assert_frame_equal(first[name],second[name])
        groups=[set(part.duplicate_group) for part in first.values()]
        self.assertTrue(groups[0].isdisjoint(groups[1]) and groups[0].isdisjoint(groups[2]) and groups[1].isdisjoint(groups[2]))
        self.assertEqual(sum(map(len,first.values())),len(self.df))
        self.assertEqual({name:len(part) for name,part in first.items()},{'train':66,'validation':12,'test':12})
    def test_split_keeps_duplicates_together(self):
        duplicate=self.df.iloc[[0]].copy(); duplicate.loc[:,'text_id']='synthetic-copy'
        parts=split_dataset(pd.concat([self.df,duplicate],ignore_index=True))
        locations=[name for name,part in parts.items() if {'synthetic-0','synthetic-copy'}<=set(part.text_id)]
        self.assertEqual(len(locations),1)
    def test_split_rejects_conflicting_duplicate_labels(self):
        duplicate=self.df.iloc[[0]].copy(); duplicate.loc[:,'text_id']='synthetic-conflict'; duplicate.loc[:,'sentiment']='negative'
        with self.assertRaises(ValueError): split_dataset(pd.concat([self.df,duplicate],ignore_index=True))
    def test_dataset_hash_and_manifest(self):
        self.assertEqual(dataframe_sha256(self.df),dataframe_sha256(self.df.sample(frac=1,random_state=7)))
        manifest=split_manifest(self.df,split_dataset(self.df),42,.9)
        self.assertEqual(manifest['splits']['train']['rows'],66)
    def test_metrics(self):
        evaluation=metrics(self.df,self.df.sentiment.tolist())
        self.assertEqual(evaluation['accuracy'],1)
        self.assertEqual(evaluation['macro_f1'],1)
        self.assertEqual(evaluation['confusion_matrix'],[[30,0,0],[0,30,0],[0,0,30]])
        wrong=['neutral']*len(self.df)
        evaluation=metrics(self.df,wrong)
        self.assertAlmostEqual(evaluation['accuracy'],1/3)
        self.assertAlmostEqual(evaluation['macro_f1'],1/6)
        with self.assertRaises(ValueError): metrics(self.df,wrong[:-1])
    def test_shared_experiment_record(self):
        evaluation=metrics(self.df,self.df.sentiment.tolist())
        record=experiment_record(model_identifier='test',dataset_version='abc',split_identifier='validation',training_configuration={},random_seed=42,evaluation=evaluation,prediction_file='predictions.csv',runtime_seconds=.1,hardware='test')
        validate_experiment_record(record)
        broken=dict(record); broken['accuracy']=2
        with self.assertRaises(ValueError): validate_experiment_record(broken)
    def test_error_and_disparity_analysis(self):
        predicted=self.df.sentiment.tolist(); predicted[0]='negative'
        self.assertEqual(len(error_table(self.df,predicted)),1)
        self.assertEqual(error_summary(self.df,predicted)['confusions'],{'positive->negative':1})
        evaluation=metrics(self.df,predicted)
        self.assertIsNone(language_gap(evaluation)['macro_f1_gap'])
        self.assertEqual(training_sizes(10),[1,2,5,8,10])

if __name__ == '__main__': unittest.main()
