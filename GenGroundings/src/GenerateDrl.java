import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;


public class GenerateDrl {

        public HashMap<String, Integer> MapRelation2ID = null;
        public Boolean isOneTimeInference = false;

        public GenerateDrl(){

        }

        public void generateDrlFile(String fnRelationIDMap,
                                    String fnRuleType,
                                    String fnOutput) throws Exception{
            readData(fnRelationIDMap);
            readRulesAndGenDrl(fnRuleType,fnOutput);
        }

        private void readData(String fnRelationIDMap) throws Exception {
            MapRelation2ID = new HashMap<String, Integer>();

            BufferedReader reader = new BufferedReader(new InputStreamReader(
                    new FileInputStream(fnRelationIDMap), "UTF-8"));
            String line = "";
            while ((line = reader.readLine()) != null) {
                String[] tokens = line.split(" ");
                Integer iRelationID = Integer.parseInt(tokens[1]);
                String strRelation = tokens[0];
                MapRelation2ID.put(strRelation, iRelationID);
            }
            reader.close();
            System.out.println("Read data success!");
        }

        private void readRulesAndGenDrl(String fnHornRules,
                               String fnOutput) throws Exception {
            System.out.println("Start to generate Drools rules......");
            BufferedReader reader = new BufferedReader(new InputStreamReader(
                    new FileInputStream(fnHornRules), "UTF-8"));
            BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
                    new FileOutputStream(fnOutput), "UTF-8"));
            writer.write("import struct.DerivedTriple;\n" +
                    "import struct.Triple;\n" +
                    "import java.util.ArrayList;\n" +
                    "import java.util.HashMap;\n\n" +
                    "global HashMap<String,String> newFacts;\n\n");

            String line = "";
            String ruleTemplate =
                    "rule \"rule%d\" \n"   +
                            "\tsalience %d\n" +
                            "\t\twhen \n"   +
                            "\t\t\t%s\n"  +
                            "\t\tthen \n"   +
                            "\t\t\t%s\n"  +
                            "end\n";
            String ruleContent = "";

            HashMap<String, Boolean> tmpLst = new HashMap<String, Boolean>();

            if (!fnHornRules.equals("")) {
                int count = 0;
                while ((line = reader.readLine()) != null) {

                    if (!line.startsWith("?"))
                        continue;

                    String[] bodys = line.split("=>")[0].trim().split("  ");
                    String[] heads = line.split("=>")[1].trim().split("  ");

                    HashMap<String, Integer> MapVariable = new HashMap<String, Integer>();
                    if (bodys.length == 3){
                        String bEntity1 = bodys[0].replace('?','$');
                        int iFstRelation = MapRelation2ID.get(bodys[1]);
                        String bEntity2 = bodys[2].replace('?','$');

                        String hEntity1 = heads[0].replace('?','$');
                        int iSndRelation = MapRelation2ID.get(heads[1]);
                        String hEntity2 = heads[2].split("\t")[0].replace('?','$');
                        String confidence = heads[2].split("\t")[1];
                        double confi = Double.parseDouble(confidence);

                        String whenPart = "$first : DerivedTriple(%s:head, %s:tail, relation == %d)\n" +
                                "\t\t\tnot DerivedTriple(head==%s && tail==%s && relation==%d)";
                                //"\t\t\tnot DerivedTriple(head==%s && tail==%s && relation==%d && (iRuleId==-1 || iRuleId==%d) && isEqualPremises($first) )";
                                //"\t\t\tnot DerivedTriple(head==%s && tail==%s && relation==%d && isEqualPremises($first))";
                        String thenPart;
                        if(isOneTimeInference){
                            thenPart = "DerivedTriple newFact = new DerivedTriple(%s,%s,%d,%s,%d,$first);\n" +
                                    "\t\t\tnewFacts.put(newFact.getTriple().toString(),newFact.toString());";
                        }
                        else {
                            thenPart = "insert(new DerivedTriple(%s,%s,%d,%s,%d,$first));";
                        }

                        //whenPart = String.format(whenPart, bEntity1, bEntity2, iFstRelation, hEntity1, hEntity2, iSndRelation, count );
                        whenPart = String.format(whenPart, bEntity1, bEntity2, iFstRelation, hEntity1, hEntity2, iSndRelation );
                        thenPart = String.format(thenPart, hEntity1, hEntity2, iSndRelation, confi, count );

                        writer.write(String.format(ruleTemplate, count, (int)(confi*1000000), whenPart, thenPart));

                    }

                    if (bodys.length == 6){

                        String bEntity1 = bodys[0].trim().replace('?','$');
                        int iFstRelation =  MapRelation2ID.get(bodys[1].trim());
                        String bEntity2 = bodys[2].trim().replace('?','$');

                        String bEntity3 = bodys[3].trim().replace('?','$');
                        int iSndRelation =  MapRelation2ID.get(bodys[4].trim());
                        String bEntity4 = bodys[5].trim().replace('?','$');

                        String hEntity1 = heads[0].trim().replace('?','$');
                        int iTrdRelation =  MapRelation2ID.get(heads[1].trim());
                        String hEntity2 = heads[2].split("\t")[0].trim().replace('?','$');
                        String confidence = heads[2].split("\t")[1].trim();
                        double confi = Double.parseDouble(confidence);
                        String whenPart_snd = "";
                        if( bEntity3.equals(bEntity1) || bEntity3.equals(bEntity2) )
                            whenPart_snd = String.format("\t\t\t$second : DerivedTriple(head == %s, %s:tail, relation == %d)\n", bEntity3, bEntity4, iSndRelation);
                        if( bEntity4.equals(bEntity1) || bEntity4.equals(bEntity2) )
                            whenPart_snd = String.format("\t\t\t$second : DerivedTriple(%s:head, tail == %s, relation == %d)\n",bEntity3,bEntity4,iSndRelation);
                        if( whenPart_snd == "" ) {
                            throw new Exception("Neither of the two parameters in the second premise are same as the first one!");
                        }
                        //String WhenPart_trd = String.format("\t\t\tnot DerivedTriple(head==%s && tail==%s && relation==%d && (iRuleId==-1 || iRuleId==%d) && isEqualPremises($first,$second) )", hEntity1, hEntity2, iTrdRelation, count );
                        //String WhenPart_trd = String.format("\t\t\tnot DerivedTriple(head==%s && tail==%s && relation==%d && isEqualPremises($first,$second))", hEntity1, hEntity2, iTrdRelation );
                        String WhenPart_trd = String.format("\t\t\tnot DerivedTriple(head==%s && tail==%s && relation==%d)", hEntity1, hEntity2, iTrdRelation );
                        String whenPart_fst = "$first : DerivedTriple(%s:head, %s:tail, relation == %d)\n";
                        String whenPart = String.format(whenPart_fst, bEntity1, bEntity2, iFstRelation) + whenPart_snd + WhenPart_trd;

                        String thenPart;
                        if(isOneTimeInference){
                            thenPart = "DerivedTriple newFact = new DerivedTriple(%s,%s,%d,%s,%d,$first,$second);\n" +
                                    "\t\t\tnewFacts.put(newFact.getTriple().toString(),newFact.toString());";
                        }
                        else {
                            thenPart = "insert(new DerivedTriple(%s,%s,%d,%s,%d,$first,$second));";
                        }
                        thenPart = String.format(thenPart, hEntity1, hEntity2, iTrdRelation, confi, count);

                        writer.write(String.format(ruleTemplate, count, (int)(confi*1000000), whenPart, thenPart));

                        writer.flush();
                    }

                    if (bodys.length == 9){

                        String bEntity1 = bodys[0].trim().replace('?','$');
                        int iFstRelation =  MapRelation2ID.get(bodys[1].trim());
                        String bEntity2 = bodys[2].trim().replace('?','$');

                        String bEntity3 = bodys[3].trim().replace('?','$');
                        int iSndRelation =  MapRelation2ID.get(bodys[4].trim());
                        String bEntity4 = bodys[5].trim().replace('?','$');

                        String bEntity5 = bodys[6].trim().replace('?','$');
                        int iTrdRelation =  MapRelation2ID.get(bodys[7].trim());
                        String bEntity6 = bodys[8].trim().replace('?','$');

                        String hEntity1 = heads[0].trim().replace('?','$');
                        int iFouRelation =  MapRelation2ID.get(heads[1].trim());
                        String hEntity2 = heads[2].split("\t")[0].trim().replace('?','$');
                        String confidence = heads[2].split("\t")[1].trim();
                        double confi = Double.parseDouble(confidence);
                        String whenPart_snd = "";

                        ArrayList<String> vars = new ArrayList<>();
                        vars.add(bEntity1);
                        vars.add(bEntity2);
                        if( vars.contains(bEntity3) ) {
                            if( vars.contains(bEntity4) )
                                whenPart_snd = String.format("\t\t\t$second : DerivedTriple(head == %s, tail == %s, relation == %d)\n", bEntity3, bEntity4, iSndRelation);
                            else
                                whenPart_snd = String.format("\t\t\t$second : DerivedTriple(head == %s, %s:tail, relation == %d)\n", bEntity3, bEntity4, iSndRelation);
                        }
                        else if( vars.contains(bEntity4) )
                            whenPart_snd = String.format("\t\t\t$second : DerivedTriple(%s:head, tail == %s, relation == %d)\n", bEntity3, bEntity4, iSndRelation);
                        else
                            whenPart_snd = String.format("\t\t\t$second : DerivedTriple(%s:head, %s:tail, relation == %d)\n", bEntity3, bEntity4, iSndRelation);

                        vars.add(bEntity3);
                        vars.add(bEntity4);
                        if( vars.contains(bEntity5) ){
                            if( vars.contains(bEntity6) )
                                whenPart_snd += String.format("\t\t\t$third : DerivedTriple(head == %s, tail == %s, relation == %d)\n", bEntity5, bEntity6, iTrdRelation);
                            else
                                whenPart_snd += String.format("\t\t\t$third : DerivedTriple(head == %s, %s:tail, relation == %d)\n", bEntity5, bEntity6, iTrdRelation);
                        }
                        else if( vars.contains(bEntity6) )
                            whenPart_snd += String.format("\t\t\t$third : DerivedTriple(%s:head, tail == %s, relation == %d)\n", bEntity5, bEntity6, iTrdRelation);
                        else
                            whenPart_snd += String.format("\t\t\t$third : DerivedTriple(%s:head, %s:tail, relation == %d)\n", bEntity5, bEntity6, iTrdRelation);


                        //String WhenPart_trd = String.format("\t\t\tnot DerivedTriple(head==%s && tail==%s && relation==%d )", hEntity1, hEntity2, iFouRelation );
                        //String WhenPart_trd = String.format("\t\t\tnot DerivedTriple(head==%s && tail==%s && relation==%d && isEqualPremises($first,$second))", hEntity1, hEntity2, iTrdRelation );
                        String WhenPart_trd = String.format("\t\t\tnot DerivedTriple(head==%s && tail==%s && relation==%d)", hEntity1, hEntity2, iTrdRelation );
                        String whenPart_fst = "$first : DerivedTriple(%s:head, %s:tail, relation == %d)\n";
                        String whenPart = String.format(whenPart_fst, bEntity1, bEntity2, iFstRelation) + whenPart_snd + WhenPart_trd;

                        String thenPart;
                        if(isOneTimeInference){
                            thenPart = "DerivedTriple newFact = new DerivedTriple(%s,%s,%d,%s,%d,$first,$second,$third);\n" +
                                    "\t\t\tnewFacts.put(newFact.getTriple().toString(),newFact.toString());";
                        }
                        else{
                            thenPart = "insert(new DerivedTriple(%s,%s,%d,%s,%d,$first,$second,$third));";
                        }
                        thenPart = String.format(thenPart, hEntity1, hEntity2, iFouRelation, confi, count);

                        writer.write(String.format(ruleTemplate, count, (int)(confi*1000000), whenPart, thenPart));

                        writer.flush();
                    }
                    count ++;

                }
                reader.close();
                writer.close();
            }
            System.out.println("Success!");
        }

        /*
        public static void main(String[] args) throws Exception {
            // TODO Auto-generated method stub
            String fnRelationIDMap = "..\\Datasets\\fb15k\\r2id.txt";
            String fnRuleType = "..\\Datasets\\fb15k\\fb15k_rule_80.txt";
            String fnOutput = "..\\Datasets\\fb15k\\groundings_80.drl";

            long startTime = System.currentTimeMillis();
            GenerateDrl generator = new GenerateDrl();
            generator.isOneTimeInference = true;
            generator.generateDrlFile(fnRelationIDMap,fnRuleType,fnOutput);

            long endTime = System.currentTimeMillis();
            System.out.println("All running time:" + (endTime-startTime)+"ms");
        }
        */
}
