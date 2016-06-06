
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
		
		ataque.setClassIndex(1); // define o �ndice do atributo classe
		
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
		
	// Gera um arquivo CSV para facilitar a tabula��o no Excel
		System.out.println("real;knn;vizinho"); // r�tulo das colunas
		
		for (int j = 0; j < iteracoes; j++ ) {
		Instances ataqueTreino = ataque.trainCV(iteracoes, j); // obt�m 2/3 do conjunto ataque 
		Instances ataqueTeste = ataque.testCV(iteracoes, j); // obt�m 1/3 do conjunto ataque
		
		// Cria os classificadores
		IBk knn = new IBk(3); // k = 3
		IB1 vizinho = new IB1();
		
		// Treina os classificadores
		knn.buildClassifier(ataqueTreino);
		vizinho.buildClassifier(ataqueTreino);
		
		
		for (int i = 0; i < ataqueTeste.numInstances(); i++) {
			Instance teste = ataqueTeste.instance(i); // obt�m um exemplo do conjunto
			System.out.print(teste.value(1) + ";"); // Imprime r�tulo original do exemplo
			teste.setClassMissing(); // Remove o r�tulo do exemplo
			// Submete exemplo � avalia��o dos classificadores
			double knnValue = knn.classifyInstance(teste);
			double vizinhoValue = vizinho.classifyInstance(teste);
			
			// Finaliza linha com os resultados dos classificadores
			System.out.println(knnValue + ";" + vizinhoValue);
		}
	}
	}


}

// procurar metodo que ignora valores com ??