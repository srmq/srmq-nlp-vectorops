package br.cin.ufpe.nlp.vectorops;

import java.util.Map;
import java.util.WeakHashMap;

import br.cin.ufpe.nlp.api.vectors.VectorVocab;

public class CachedVectorVocab implements VectorVocab {
	private VectorVocab wrappedIndex;
	private Map<String, float[]> cache;
	private boolean flushed;
	private int cacheSize;
	
	public CachedVectorVocab(VectorVocab index, int cacheSize) {
		this.cacheSize = cacheSize;		
		this.wrappedIndex = index;
		allocateCache(cacheSize);
	}

	private void allocateCache(int cacheSize) {
		this.cache = new WeakHashMap<String, float[]>(cacheSize);
		flushed = false;
	}
	
	public synchronized float[] embeddingFor(String word) {
		if (flushed) allocateCache(this.cacheSize);
		float[] ret = cache.get(word);
		if (ret == null) {
			ret = wrappedIndex.embeddingFor(word);
			if (ret != null) {
				cache.put(word, ret);
			}
		}
		
		return ret;
	}
	
	public synchronized boolean contains(String word) {
		if (flushed) allocateCache(this.cacheSize);
		boolean ret = false;
		if (cache.containsKey(word)) {
			ret = true;
		} else {
			final float[] embed = wrappedIndex.embeddingFor(word);
			if (embed != null) {
				cache.put(word, embed);
				ret = true;
			}
		}
		return ret;
	}
	
	public synchronized void flush() {
		flushed = true;
		this.cache = null;
	}

	@Override
	public int numWords() {
		return wrappedIndex.numWords();
	}

	@Override
	public int embedSize() {
		return wrappedIndex.embedSize();
	}

}
