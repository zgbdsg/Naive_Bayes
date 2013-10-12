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
	
    public NaiveBayes() {
    }

    @Override
    public void train(boolean[] isCategory, double[][] features, double[] labels) {
    	prob = new dataProbability[features[0].length];
    	proby = new dataProbability(true, labels, null);
    	
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
    			
    			for(int k=0;k<datax.length;k ++){
    				if((double)datax[k] == features[j]){
    					probability = probability*prob[j].probability[i*datax.length+k];
    					break;
    				}
    			}
    		}
    		
    		if(probability > max) {
    			max = probability;
    			label = (double)proby.datax[i];
    		}

    	}
        return label;
    }
}

class dataProbability{
	Object[] datax;  //different data in x
	Object[] datay;  //different data in y
	boolean isCategory;  // type of x
	double[] probability;
	
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
				this.probability[i] = 1.0*info.amount[i] / total;
			}
			
		}else {
			//compute conditionprobability
			FeatureInfo labelInfo = process_feature(true, label);
			FeatureInfo featureInfo = process_feature(isCategory, feature);
			
			this.datax = featureInfo.feature.toArray();
			this.datay = labelInfo.feature.toArray();
			this.probability = new double[featureInfo.numType*labelInfo.numType];
			
			if(isCategory) {
				//feature is category
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
						
						probability[i*featureInfo.numType + j] = 1.0*tmp / labeltotal;
					}
				}
			}else{
				//feature is numeric
				FeatureInfo numericinfo = process_feature(true, featureInfo.feature.toArray());
				
				List<Object> tmpfea = featureInfo.feature;
				
				for(int i=0;i < labelInfo.numType;i ++) {
					double labeldata = (double)datay[i];
					double labeltotal = labelInfo.amount[i];
					for(int j=0;j < numericinfo.numType;j ++) {
						int tmp = 0;
						double featuredata = featureInfo.amount[tmpfea.indexOf(numericinfo.feature.get(j))];
						for(int k=0;k < feature.length;k ++) {
							if((double)feature[k] <= featuredata && (double)label[k] == labeldata)
								tmp ++;
						}
						
						probability[i*featureInfo.numType + j] = 1.0*tmp / labeltotal;
					}
				}
			}
		}

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