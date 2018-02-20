package br.cin.ufpe.nlp.vectorops;

import java.io.File;

import br.cin.ufpe.nlp.api.vectors.VectorVocab;
import br.cin.ufpe.nlp.api.vectors.VectorVocabService;

public class LuceneVectorVocabService implements VectorVocabService {

	@Override
	public VectorVocab loadVectorVocab(File vectorFile, String indexPath, boolean normalizeVectors) {
		LuceneBackedVectorVocab luceneVocab = new LuceneBackedVectorVocab(vectorFile, indexPath, true, normalizeVectors);
		CachedVectorVocab cachedVocab = new CachedVectorVocab(luceneVocab, 50000);
		return cachedVocab;
	}

}
