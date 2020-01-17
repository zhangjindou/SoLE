package struct;

public class Triple {
	private int iHeadEntity;
	private int iTailEntity;
	private int iRelation;
	
	public Triple() {
	}
	
	public Triple(int i, int j, int k) {
		iHeadEntity = i;
		iTailEntity = j;
		iRelation = k;
	}
	
	public int head() {
		return iHeadEntity;
	}
	
	public int tail() {
		return iTailEntity;
	}
	
	public int relation() {
		return iRelation;
	}

	@Override
	public String toString(){
		return String.format("%d_%d_%d",iHeadEntity,iRelation,iTailEntity);
	}

	@Override
	public boolean equals(Object obj) {
		if( obj instanceof Triple){
			Triple t = (Triple)obj;
			return t.head()==this.iHeadEntity && t.tail()==this.iTailEntity && t.relation()==this.iRelation;
		}
		return false;
	}
}
