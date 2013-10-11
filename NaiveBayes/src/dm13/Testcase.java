package dm13;

import java.util.Arrays;

import auxiliary.DataSet;
import auxiliary.NaiveBayes;

public class Testcase {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		DataSet dataset = new DataSet("breast-cancer.data");
		//DataSet dataset = new DataSet("segment.data");
		NaiveBayes nb = new NaiveBayes();
		
		double[][] data = dataset.getFeatures();
		 double[] test = new double[data.length];
		 for(int i=0;i < data.length;i ++) {
			 System.out.println(Arrays.toString(data[i]));
			 test[i] = data[i][7];
		 }
		
		nb.train(dataset.getIsCategory(), dataset.getFeatures(), dataset.getLabels());
	}

}
