


import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.text.DecimalFormat;


public class Rico {


    public static void main(String[] args) throws Exception {

        ConverterUtils.DataSource ds=new ConverterUtils.DataSource("src/base.arff");

         Instances ins = ds.getDataSet();
         ins.setClassIndex(7);
         NaiveBayes nb =new NaiveBayes();
         nb.buildClassifier(ins);
         Instance novo=new Instance(8);
         novo.setDataset(ins);
         novo.setValue(0,3);
         novo.setValue(1,"Vendas");
         novo.setValue(2,7);
         novo.setValue(3,"Financiada");
         novo.setValue(4,1000);
        novo.setValue(5,2);
        novo.setValue(6,2000);

         double probabilidade[]=nb.distributionForInstance(novo);
         float dfa =Math.round(probabilidade[0]*100);
         float dfb=Math.round(probabilidade[1]*100);
        System.out.println("probabilidade de pagar o emprestimo="+ " " + "%" + dfa);
        System.out.println("probabilidade de  nao pagar emprestimo"+" "+ "%"+dfb);
    }







}
