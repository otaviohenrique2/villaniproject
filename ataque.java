
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

import weka.classifiers.lazy.IB1;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

public class ataque {

	public static void main(String[] args) throws Exception {
		// Cria um conjunto de dados (Instances) a partir do ARFF
		FileReader leitor = new FileReader("ataque.arff");
		Instances ataque = new Instances(leitor);
		
		ataque.setClassIndex(1); // define o índice do atributo classe
		
		ataque = ataque.resample(new Random()); // embaralha a ordem dos exemplos no conjunto

		int iteracoes = ataque.numInstances();
		
		BufferedReader reader =
			   new BufferedReader(new FileReader("ataque.arff"));
			 ArffReader arff = new ArffReader(reader, 1000);
			 Instances data = arff.getStructure();
			 data.setClassIndex(data.numAttributes() - 1);
			 Instance inst;
			 while ((inst = arff.readInstance(data)) != null) {
			   data.add(inst);
			 }
		
	// Gera um arquivo CSV para facilitar a tabulação no Excel
		System.out.println("real;knn;vizinho"); // rótulo das colunas
		
		for (int j = 0; j < iteracoes; j++ ) {
		Instances ataqueTreino = ataque.trainCV(iteracoes, j); // obtém 2/3 do conjunto ataque 
		Instances ataqueTeste = ataque.testCV(iteracoes, j); // obtém 1/3 do conjunto ataque
		
		// Cria os classificadores
		IBk knn = new IBk(3); // k = 3
		IB1 vizinho = new IB1();
		
		// Treina os classificadores
		knn.buildClassifier(ataqueTreino);
		vizinho.buildClassifier(ataqueTreino);
		
		
		for (int i = 0; i < ataqueTeste.numInstances(); i++) {
			Instance teste = ataqueTeste.instance(i); // obtém um exemplo do conjunto
			System.out.print(teste.value(1) + ";"); // Imprime rótulo original do exemplo
			teste.setClassMissing(); // Remove o rótulo do exemplo
			// Submete exemplo à avaliação dos classificadores
			double knnValue = knn.classifyInstance(teste);
			double vizinhoValue = vizinho.classifyInstance(teste);
			
			// Finaliza linha com os resultados dos classificadores
			System.out.println(knnValue + ";" + vizinhoValue);
		}
	}
	}


}

// procurar metodo que ignora valores com ??