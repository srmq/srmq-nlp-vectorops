package br.cin.ufpe.nlp.vectorops;

import java.util.Hashtable;

import org.osgi.framework.BundleActivator;
import org.osgi.framework.BundleContext;

import br.cin.ufpe.nlp.api.vectors.VectorVocabService;

public class Activator implements BundleActivator {

	@Override
	public void start(BundleContext context) throws Exception {
		Hashtable<String, String> props = new Hashtable<String, String>();
		context.registerService(VectorVocabService.class.getName(), new LuceneVectorVocabService(), props);
	}

	@Override
	public void stop(BundleContext arg0) throws Exception {
		// NOTE: The service is automatically unregistered.
	}

}
