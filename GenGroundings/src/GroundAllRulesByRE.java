import struct.DerivedTriple;
import org.kie.api.KieServices;
import org.kie.api.builder.KieBuilder;
import org.kie.api.builder.KieFileSystem;
import org.kie.api.builder.KieRepository;
import org.kie.api.io.Resource;
import org.kie.api.io.ResourceType;
import org.kie.api.runtime.KieContainer;
import org.kie.api.runtime.KieSession;
import org.kie.api.runtime.rule.FactHandle;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;


public class GroundAllRulesByRE {

    private ArrayList<DerivedTriple> aTrainingData = null;
    private ArrayList<DerivedTriple> aInferredFacts = new ArrayList<>();
    private ArrayList<DerivedTriple> aGroundingFacts = new ArrayList<>();


    public GroundAllRulesByRE(String fTrainingTriples) throws Exception{
        aTrainingData = new ArrayList<DerivedTriple>();
        readData(fTrainingTriples);
    }

    private void readData( String fTriples ) throws Exception{
        BufferedReader reader = new BufferedReader(new InputStreamReader(
                new FileInputStream(fTriples), "UTF-8"));
        String line = "";
        while ((line = reader.readLine()) != null) {
            String[] tokens = line.split(",");
            Integer iSubject = Integer.parseInt(tokens[0]);
            Integer iRelationID = Integer.parseInt(tokens[1]);
            Integer iObject = Integer.parseInt(tokens[2]);
            DerivedTriple tri = new DerivedTriple(iSubject,iObject,iRelationID);
            aTrainingData.add(tri);
        }
        reader.close();
        System.out.println(String.format("Read %d training triples successfully!",aTrainingData.size()));
    }


    public ArrayList<String> inferUnlabeledTriples( String fnDrlFile ) throws Exception{

        KieServices ks = KieServices.Factory.get();
        KieRepository kr = ks.getRepository();
        KieFileSystem kfs = ks.newKieFileSystem();
        HashMap<String,String> oneTimeInferedFacts = new HashMap<>();

        File file = new File(fnDrlFile);
        Resource resource = ks.getResources().newFileSystemResource(file).setResourceType(ResourceType.DRL);
        kfs.write( resource );


        KieBuilder kb = ks.newKieBuilder(kfs);

        kb.buildAll(); // kieModule is automatically deployed to KieRepository if successfully built.
//        if (kb.getResults().hasMessages(Message.Level.ERROR)) {
//            throw new RuntimeException("Build Errors:\n" + kb.getResults().toString());
//        }

        KieContainer kContainer = ks.newKieContainer(kr.getDefaultReleaseId());

        KieSession kSession = kContainer.newKieSession();
        kSession.setGlobal("newFacts", oneTimeInferedFacts);

        for ( DerivedTriple triple : aTrainingData ) {
            kSession.insert( triple );
        }

        System.out.println("Fire All Rules...");
        kSession.fireAllRules();
        for( FactHandle fact: kSession.getFactHandles() ){
            DerivedTriple dt = (DerivedTriple) kSession.getObject(fact);

            if( dt.getaPremiseTriples() != null && dt.getaPremiseTriples().size()>0 ) {
                aInferredFacts.add(dt);
            }
        }
        System.out.println(String.format("After firing, there are %d triples",kSession.getFactCount() ));
        kSession.dispose();

        return new ArrayList<>(oneTimeInferedFacts.values());
    }

    public ArrayList<DerivedTriple> getInferredFacts() {
        return aInferredFacts;
    }

    public ArrayList<DerivedTriple> getGroundings() {
        return aGroundingFacts;
    }

    public static void main(String[] args) throws Exception {

        String[] confidenceLevels =  {"80"}; // {"60","65","70","75","80","85","90","95","100"};
        String path_prefix = "../Datasets/fb15k/"; //"datasets\\wn18\\";

        //Convert the rule mined by AMIE+ to Drool rule
        GenerateDrl generator = new GenerateDrl();
        generator.isOneTimeInference = false;  // perform reasoning by forward chaining or one-time inference.
        String fnRelationIDMap = path_prefix + "r2id.txt"; // "\\r2id.txt";
        String fnEntityIDMap = path_prefix + "e2id.txt"; // "\\e2id.txt";

        for(int i=0; i<confidenceLevels.length; i++) {
            String fnRuleType = path_prefix + String.format("rule_%s.txt", confidenceLevels[i]);
            String fnDroolRule = path_prefix + String.format("groundings_%s.drl", confidenceLevels[i]);

            long startTime = System.currentTimeMillis();
            generator.generateDrlFile(fnRelationIDMap,fnRuleType,fnDroolRule);

            long endTime = System.currentTimeMillis();
            System.out.println("All running time:" + (endTime-startTime)+"ms");
        }


        //Perform reasoning
        String fnTrainingTriples = path_prefix + "digitized_train.txt";
        //String fn_Ent = path_prefix + "e2id.txt"; // "\\e2id.txt";
        //String fn_Rel = path_prefix + "r2id.txt"; // "\\r2id.txt";
        BufferedReader read = new BufferedReader(new InputStreamReader(
                new FileInputStream(fnEntityIDMap), "UTF-8"));
        int entitiesNum = 0, relationsNum = 0;
        while (read.readLine() != null) {
            entitiesNum++;
        }
        read.close();
        read = new BufferedReader(new InputStreamReader(
                new FileInputStream(fnRelationIDMap), "UTF-8"));
        while (read.readLine() != null) {
            relationsNum++;
        }
        read.close();
        System.out.println(String.format("entities: %d, relations: %d", entitiesNum, relationsNum));

        for(int i=0; i<confidenceLevels.length; i++) {
            long startTime = System.currentTimeMillis();
            String fnGroundRules = path_prefix + "groundings_" + confidenceLevels[i] + ".drl";
            //check whether drools rule file exists
            File f = new File(fnGroundRules);
            if( !f.exists() )
                continue;
            ArrayList<String> oneTimeInferedFacts;
            GroundAllRulesByRE grounding = new GroundAllRulesByRE(fnTrainingTriples);
            oneTimeInferedFacts = grounding.inferUnlabeledTriples(fnGroundRules);

            BufferedWriter write_grouningsRE = new BufferedWriter(new OutputStreamWriter(
                    new FileOutputStream(path_prefix + "groundings_" + confidenceLevels[i] + ".txt"), "UTF-8"));

//        BufferedWriter write_grounings = new BufferedWriter(new OutputStreamWriter(
//                new FileOutputStream("./datasets/fb15k/groundings.txt"), "UTF-8"));

            for (DerivedTriple dt : grounding.getInferredFacts()) {
                write_grouningsRE.write(dt.toString().replaceAll("0 0 0",
                        String.format("%d %d %d", entitiesNum, relationsNum, entitiesNum)) + "\n");
            }
//        for(DerivatedTriple dt: grounding.getGroundings()){
//            write_grounings.write(dt.toString()+"\n");
//        }
            write_grouningsRE.close();
//        write_grounings.close();
            BufferedWriter write_groundingsOneTime = new BufferedWriter(new OutputStreamWriter(
                    new FileOutputStream(path_prefix + "groundings_oneTime_"+ confidenceLevels[i] + ".txt"), "UTF-8"));
            for (String t : oneTimeInferedFacts) {
                write_groundingsOneTime.write(t.replaceAll("0 0 0",
                        String.format("%d %d %d", entitiesNum, relationsNum, entitiesNum)) + "\n");
            }
            write_groundingsOneTime.flush();
            write_groundingsOneTime.close();

            long endTime = System.currentTimeMillis();
            System.out.println(fnGroundRules + " running time:" + (endTime-startTime)+"ms");
        }

    }

}
