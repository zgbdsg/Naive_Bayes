/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package auxiliary;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 *
 * @author zhougb
 */
public class NaiveBayes extends Classifier {

	dataProbability[] prob;
	dataProbability proby;
	boolean[] isCategory;
	
    public NaiveBayes() {
    }

    @Override
    public void train(boolean[] isCategory, double[][] features, double[] labels) {
    	prob = new dataProbability[features[0].length];
    	proby = new dataProbability(true, labels, null);
    	this.isCategory = isCategory;
    	
    	//System.out.println(proby.toString());
    	
    	double[][] tranFeature = new double[features[0].length][features.length];
    	for(int i=0;i<features[0].length;i ++){
    		for(int j=0;j<features.length;j ++)
    			tranFeature[i][j] = features[j][i];
    	}
    	
    	for(int i=0;i< tranFeature.length;i ++) {
    		prob[i] = new dataProbability(isCategory[i], tranFeature[i], labels);
    		//System.out.println(prob[i].toString());
    	}
    	
    	//System.out.println(prob[0].result[0][0]+"    "+prob[0].result[0][1]);
    }

    @Override
    public double predict(double[] features) {
    	//Object[] ofea = new Object[features.length];
    	//FeatureInfo info = dataProbability.process_feature(true, features);
    	
    	Object[] datay = prob[0].datay;
    	double label=0;
    	double max = 0;
    	
    	for(int i=0;i<datay.length;i ++) {
    		double probability = proby.probability[i];
    		
    		for(int j=0;j<features.length;j ++){
    			Object[] datax = prob[j].datax;
    			double[][] result = prob[j].result;
    			
    			if(this.isCategory[j]) {
	    			for(int k=0;k<datax.length;k ++){
	    				if((double)datax[k] == features[j]){
	    					probability = probability*prob[j].probability[i*datax.length+k];
	    					break;
	    				}
	    			}
    			}else{
    				//System.out.println(result[0][0]+"    "+result[0][1]);
    				//System.out.println(prob[0].result[0][0]+"    "+prob[0].result[0][1]);
    				double average = result[i][0];
    				double variance = result[i][1];
    				
    				if(variance > 0.0000000001){
	    				double a = (Math.sqrt(2.0*Math.PI)*variance);
	    				double b = (features[j]-average)*(features[j]-average);
	    				double c = 2.0*variance*variance;
	    				double d = (1.0*(1.0/a)*Math.exp(-1.0*b/c));
	    				probability = probability*d;
    				}else{
    					if(Math.abs(average-features[j]) < 0.0000000001){
    						probability = probability;
    					}else{
    						probability = 0;
    					}
    				}
    			}
    		}
    		
    		if(probability > max) {
    			max = probability;
    			label = (double)proby.datax[i];
    		}

    	}
    	//System.out.println(Arrays.toString(features));
    	//System.out.println(label);
        return label;
    }
}



class dataProbability{
	Object[] datax;  //different data in x
	Object[] datay;  //different data in y
	boolean isCategory;  // type of x
	double[] probability;
	double[][] result;
	
	public dataProbability(boolean isCategory,double[] datax,double[] datay) {
		this.isCategory = isCategory;
		Object[] dataA = new Object[datax.length];
		Object[] dataB;
		if(datay != null)
			dataB = new Object[datay.length];
		else
			dataB = null;
		
		for(int i=0;i<datax.length;i ++) {
			dataA[i] = datax[i];
			if(datay != null)
				dataB[i] = datay[i];
		}
		setProbability(isCategory, dataA, dataB);
	}
	
	public void setProbability(boolean isCategory,Object[] feature,Object[] label) {
		
		if(label == null) {
			//compute prioriprobability
			FeatureInfo info = process_feature(true, feature);
			
			this.datax = info.feature.toArray();
			this.probability = new double[info.numType];
			int total = 0;
			for(int i=0;i < info.numType;i ++) {
				total += info.amount[i];
			}
			
			for(int i=0;i < info.numType;i ++) {
				this.probability[i] = 1.0*(info.amount[i]+1.0) / (total+1.0*this.datax.length);
			}
			
		}else {
			//compute conditionprobability
			FeatureInfo labelInfo = process_feature(true, label);
			this.datay = labelInfo.feature.toArray();
			
			if(isCategory) {
				//feature is category
				FeatureInfo featureInfo = process_feature(isCategory, feature);
				
				this.datax = featureInfo.feature.toArray();
				
				this.probability = new double[featureInfo.numType*labelInfo.numType];
				for(int i=0;i < labelInfo.numType;i ++) {
					double labeldata = (double)datay[i];
					double labeltotal = labelInfo.amount[i];
					for(int j=0;j < featureInfo.numType;j ++) {
						int tmp = 0;
						double featuredata = (double)datax[j];
						for(int k=0;k < feature.length;k ++) {
							if((double)feature[k] == featuredata && (double)label[k] == labeldata)
								tmp ++;
						}
						
						probability[i*featureInfo.numType + j] = 1.0*(tmp+1.0) / (labeltotal+1.0*featureInfo.numType);
					}
				}
			}else{
				//feature is numeric

				double[][] newfea = new double[labelInfo.numType][];
				this.result = new double[labelInfo.numType][2];
				
				for(int i=0;i < labelInfo.numType;i ++) {
					newfea[i] = new double[labelInfo.amount[i]];
				}
				
				for(int i=0;i < labelInfo.numType;i ++) {
					int num=0;
					for(int j=0;j<label.length;j ++) {
						if((double)label[j] == (double)labelInfo.feature.get(i)){
							newfea[i][num ++] = (double)feature[j];
						}
					}
				}
				
				for(int i=0;i<labelInfo.numType;i ++) {
					this.result[i] = get_Gauss(newfea[i]);
				}
			}
		}

	}
	
	public double[] get_Gauss(double[] feature) {
		
		double[] result = new double[2];
		double average = 0;
		double variance = 0;
		
		double total = 0;
		for(int i=0;i<feature.length;i ++){
			total += feature[i];
		}
		average = 1.0*total / feature.length;
		
		for(int i=0;i< feature.length;i ++) {
			variance += Math.pow((feature[i]-average), 2);
		}
		variance = Math.sqrt(variance / feature.length);
		result[0] = average;
		result[1] = variance;
		return result;
	}
	
	public static FeatureInfo process_feature(boolean isCategory, Object[] data) {
			// find the not repeat element , its amount
			int numType = 0;
			List<Object> feature = new ArrayList<Object>();
			int[] amount;

			if (isCategory) {
				for (int i = 0; i < data.length; i++) {
					if (!feature.contains((Object) data[i])) {
						if (!Double.isNaN((double)data[i])) {
							numType++;
							feature.add(data[i]);
						}

					}
				}

				amount = new int[numType];

				for (int i = 0; i < data.length; i++) {
					int k = feature.indexOf((Object) data[i]);
					if (k >= 0)
						amount[k]++;
				}
			} else {
				Arrays.sort(data);
				List<Object> tmp = new ArrayList<Object>();
				for (int i = 0; i < data.length; i++) {
					if (!feature.contains((Object) data[i])) {
						if (!Double.isNaN((double)data[i])) {
							numType++;
							tmp.add(data[i]);
						}
					}
				}

				for (int i = 0; i < tmp.size() - 1; i++) {
					feature.add((Object) (1.0 * ((double) tmp.get(i) + (double) tmp
							.get(i + 1)) / 2.0));
				}

				numType = numType - 1;
				if (numType > 0) {
					amount = new int[numType];

					if (isCategory) {
						for (int i = 0; i < data.length; i++) {
							int k = feature.indexOf((Object) data[i]);
							if (k >= 0)
								amount[k]++;
						}
					} else {

						for (int k = 0; k < feature.size(); k++) {
							for (int i = 0; i < data.length; i++) {
								if ((double)data[i] <= (double) feature.get(k)) {
									amount[k]++;
								} else
									break;
							}
						}
					}
				} else
					amount = null;
			}
			// System.out.println(numType);
			// System.out.println(Arrays.toString(amount));
			FeatureInfo info = new FeatureInfo(numType, feature, amount);
			//System.out.println(info.toString());
			return info;
		}

	@Override
	public String toString() {
		return "dataProbability [datax=" + Arrays.toString(datax) + ", datay="
				+ Arrays.toString(datay) + ", isCategory=" + isCategory
				+ ", probability=" + Arrays.toString(probability) + "]";
	}
	
}

class FeatureInfo {
	int numType;
	List<Object> feature;
	int[] amount;

	public FeatureInfo(int numType, List<Object> feature, int[] amount) {
		this.numType = numType;
		if (numType > 0) {
			this.feature = feature;
		} else
			this.feature = null;

		this.amount = amount;
	}

	@Override
	public String toString() {
		return "FeatureInfo [numType=" + numType + ", feature="
				+ feature.toString() + ", amount="
				+ Arrays.toString(amount) + "]";
	}

}