package cz.zcu.kiv.nlp.ir.trec.data;

/**
 * Created by Tigi on 6.1.2015.
 *
 * Třída {@link AbstractResult} implementuje rozhraní {@link Result}
 *
 * Představuje výsledek pro ohodnocené vyhledávání. Tzn. po zadání dotazu vyhledávač vrátí
 * "List<Result>", kde každý objekt {@link Result} reprezentuje jeden dokument a jeho relevanci k zadanému dotazu.
 *  => tj. id dokumentu, skóre podobnosti mezi tímto dokumentem a dotazem (např. kosinova podobnost), a rank tj.
 *  pořadí mezi ostatními vrácenými dokumenty (dokument s rankem 1 bude dokument, který je nejrelevantnější k dodtazu)
 *
 * Od této třídy byste měli dědit pokud vám nestačí implementace třídy {@link ResultImpl}, např. pokud potřebujete
 * přidat nějaké další proměnné.
 *
 * Metodu toString(String topic) neměnte, ani nepřepisujte v odděděných třídách slouží pro generování výstupu
 * v daném formátu pro vyhodnocovací skript.
 *
 */
public abstract class AbstractResult implements Result {

    /**
     * Id dokumentu
     */
    String documentID;

    /**
     * Rank (pořadí) mezi ostatními vrácenými dokumenty
     */
    int rank = -1;

    /**
     * Skóre podobnosti mezi tímto výsledkem (dokumentem) a dotazem
     */
    float score = -1;

    @Override
    public String getDocumentID() {
        return documentID;
    }

    @Override
    public float getScore() {
        return score;
    }

    public void setDocumentID(String documentID) {
        this.documentID = documentID;
    }

    public void setRank(int rank) {
        this.rank = rank;
    }

    @Override
    public int getRank() {
        return rank;
    }

    public void setScore(float score) {
        this.score = score;
    }

    @Override
    public String toString() {
        return "Result{" +
                "documentID='" + documentID + '\'' +
                ", rank=" + rank +
                ", score=" + score +
                '}';
    }

    /**
     * Metoda používaná pro generování výstupu pro vyhodnocovací skript.
     * Metodu nepřepisujte (v potomcích) ani neupravujte
     */
    @Override
    public String toString(String topic) {
        return topic + " Q0 " + documentID + " " + rank + " " + score + " runindex1";
    }
}
