package struct;

import java.util.ArrayList;

public class DerivedTriple {
    private int head;
    private int tail;
    private int relation;

    private ArrayList<DerivedTriple> aPremiseTriples = null;
    private double dConfidence = 1.0;

    private int iRuleId = -1;

    public DerivedTriple(int h, int t, int r){
        head = h;
        tail = t;
        relation = r;
    }


    public DerivedTriple(int h, int t, int r, double conf, DerivedTriple... premises){
        head = h;
        tail = t;
        relation = r;

        dConfidence = conf;
        aPremiseTriples = new ArrayList<>();
        for(DerivedTriple item:premises)
            aPremiseTriples.add(item);

    }
    public DerivedTriple(int h, int t, int r, double conf,  int ruleid, DerivedTriple... premises){
        head = h;
        tail = t;
        relation = r;

        dConfidence = conf;
        iRuleId = ruleid;
        aPremiseTriples = new ArrayList<>();
        for(DerivedTriple item:premises)
            aPremiseTriples.add(item);

    }

    public double getdConfidence() {
        return dConfidence;
    }

    public ArrayList<DerivedTriple> getaPremiseTriples() {
        return aPremiseTriples;
    }

    public int getHead() {
        return head;
    }

    public int getTail() {
        return tail;
    }

    public int getRelation() {
        return relation;
    }

    public Triple getTriple(){
        Triple ret = new Triple(head,tail,relation);
        return ret;
    }

    public int getiRuleId() {
        return iRuleId;
    }

    public boolean isEqualPremises(DerivedTriple ...  premises ){
        if( aPremiseTriples==null )
            return true;
        else{
            if( aPremiseTriples.size() != premises.length )
                return false;
            for( int idx = 0; idx< aPremiseTriples.size(); idx++){
                DerivedTriple rightDt = aPremiseTriples.get(idx);
                DerivedTriple leftDt = premises[idx];
                if(!rightDt.getTriple().equals(leftDt.getTriple()))
                    return  false;
            }
        }

        return true;
    }

    public String getUniqueID(){
        String retStr = "";
        if( aPremiseTriples != null){
            for(int i=0; i<aPremiseTriples.size(); i++)
                retStr += String.format("%d_%d_%d_", aPremiseTriples.get(i).getHead(),
                        aPremiseTriples.get(i).getRelation(), aPremiseTriples.get(i).getTail());
        }
        retStr += String.format("%d_%d_%d",head,relation,tail);
        return retStr;
    }

    @Override
    public String toString() {
        if( aPremiseTriples==null )
            return String.format("-1\t1.0\t(%d %d %d)",head,relation,tail);
        else{
            String ret = "";
            ret += String.format("%d\t%s",iRuleId,dConfidence);
            int iMaxLength = 2;
            for(int i=0; i<iMaxLength; i++){
                if( aPremiseTriples.size() > i ) {
                    ret += String.format("\t(%d %d %d)", aPremiseTriples.get(i).getHead(),
                            aPremiseTriples.get(i).getRelation(), aPremiseTriples.get(i).getTail());
                }
                else
                    ret += String.format("\t(0 0 0)");
            }
            ret += String.format("\t(%d %d %d)",head,relation,tail);

            return ret;
        }


//            String ret = "";
//            ret += String.format("%d",aPremiseTriples.size()+1);
//            for(DerivatedTriple dt: aPremiseTriples){
//                ret += String.format("\t(%d\t%d\t%d)", dt.getHead(), dt.getRelation(), dt.getTail());
//            }
//
//            ret += String.format("\t(%d\t%d\t%d)\t%s",head,relation,tail,dConfidence);
//
//            return ret;

    }
}
