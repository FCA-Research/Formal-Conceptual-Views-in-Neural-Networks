(ns obstsalat
  (:require [conexp.fca.contexts :as fast]
            [conexp.io.contexts :as fast]
            [conexp.fca.fast :as fast]
            [conexp.gui.draw :refer :all]
            [conexp.layouts.base :refer :all]
            [conexp.layouts.freese :refer :all]
            [clojure.string :as string]
            [clojure.core.async :as async]))

(def r 
  (read-context
   "scales/fruit_class/resnet_16_w_bin.csv" 
   :named-binary-csv))

(defn positive-ctx
  [ctx]
  (make-context (objects ctx)
                (filter #(= "+" (str (first %))) (attributes ctx))
                (incidence ctx)))

(defn object-concept-lattice-filter
  "Computes all concepts containing object g and their covering
  concepts."
  [ctx g]
  (let [first-C [(context-object-closure ctx #{g}) 
                 (object-derivation ctx #{g})]]
    (loop [BV #{first-C} 
           queue #{first-C}]
      (if (empty? queue) 
        BV
        (let [C (first queue)]
          (let [covering-C (direct-upper-concepts ctx C)
                ;; those not containing g can be added since they are in cover with a concept containing c
                new-C (difference covering-C BV) 
                ;; only continue with those that contain g to ensure selection criteria
                ]
            (recur (into BV new-C) (into (disj queue C) new-C))))))))

(doseq [f files]
  (let [ctx (read-context f :named-binary-csv)]
    (println (last (string/split f #"/"))
             (count (fast/concepts :pcbo ctx)))))

(doseq [f files]
  (let [ctx (read-context f :named-binary-csv)
        Pos (fn [A] (filter #(= "+" (-> % first str)) A))
        ctx (make-context (objects ctx)
                          (-> ctx attributes Pos)
                          (incidence-relation ctx))]
    (println (last (string/split f #"/")) " Pos "
             (count (fast/concepts :pcbo ctx)))))

(doseq [f files32]
  (let [ctx (read-context f :named-binary-csv)]
    (println (last (string/split f #"/"))
             (count (fast/concepts :pcbo ctx)))))

(doseq [f files32]
  (let [ctx (read-context f :named-binary-csv)
        Pos (fn [A] (filter #(= "+" (-> % first str)) A))
        ctx (make-context (objects ctx)
                          (-> ctx attributes Pos)
                          (incidence-relation ctx))]
    (println (last (string/split f #"/")) " Pos "
             (count (fast/concepts :pcbo ctx)))))

(def directory 
   (clojure.java.io/file "scales/fruit_class" ))
(def files 
       (filter #(= "16_w_bin.csv" 
                   (subs % 
                         (- (count %) (count "16_w_bin.csv"))))
       (map #(.getAbsolutePath %)(file-seq directory))))
(def files32
       (filter #(= "32_w_bin.csv" 
                   (subs % 
                         (- (count %) (count "32_w_bin.csv"))))
       (map #(.getAbsolutePath %)(file-seq directory))))

(doseq [f files]
   (let [ctx (read-context f :named-binary-csv)
         ctx+ (positive-ctx ctx)]
         (println f "\n")
         (let [common #(count (intersection (set %1) (set %2)))
               cherry (future (object-concept-lattice-filter ctx "Cherry 1"))
               cherry+ (future (object-concept-lattice-filter ctx+ "Cherry 1"))
               plum (future (object-concept-lattice-filter ctx "Plum"))
               plum+ (future (object-concept-lattice-filter ctx+ "Plum"))
               pink-lady (future(object-concept-lattice-filter ctx "Apple Pink Lady"))
               pink-lady+ (future(object-concept-lattice-filter ctx+ "Apple Pink Lady"))
               apple-red (future(object-concept-lattice-filter ctx "Apple Red 1"))
               apple-red+ (future(object-concept-lattice-filter ctx+ "Apple Red 1"))]
               (println "Fruit,Filter,Cherry,Plum,Pink Lady,Apple Red")
               (let [cur @cherry+] (println "Cherry+,"(count cur)","(common cur @cherry+)","(common cur @plum+)","(common cur @pink-lady+)","(common cur @apple-red+)))
               (let [cur @plum+] (println "Plum+,"(count cur)","(common cur @cherry+)","(common cur @plum+)","(common cur @pink-lady+)","(common cur @apple-red+)))
               (let [cur @pink-lady+] (println "Pink Lady+,"(count cur)","(common cur @cherry+)","(common cur @plum+)","(common cur @pink-lady+)","(common cur @apple-red+)))
               (let [cur @apple-red+] (println "Apple Red+,"(count cur)","(common cur @cherry+)","(common cur @plum+)","(common cur @pink-lady+)","(common cur @apple-red+)))
               (let [cur @cherry] (println "Cherry,"(count cur)","(common cur @cherry)","(common cur @plum)","(common cur @pink-lady)","(common cur @apple-red)))
               (let [cur @plum] (println "Plum,"(count cur)","(common cur @cherry)","(common cur @plum)","(common cur @pink-lady)","(common cur @apple-red)))
               (let [cur @pink-lady] (println "Pink Lady,"(count cur)","(common cur @cherry)","(common cur @plum)","(common cur @pink-lady)","(common cur @apple-red)))
               (let [cur @apple-red] (println "Apple Red,"(count cur)","(common cur @cherry)","(common cur @plum)","(common cur @pink-lady)","(common cur @apple-red))))))

(def vgg16-plum+ (let [f "scales/fruit_class/vgg16_16_w_bin.csv"]
                               (let [ctx (read-context f :named-binary-csv)
                                     ctx+ (positive-ctx ctx)]
                                 (object-concept-lattice-filter ctx+ "Plum"))))

(draw-layout
     (update-valuations 
      (freese-layout 
       (make-lattice-nc 
       vgg16-plum+
        (fn [[a b] [a2 b2]] 
          (subset? a a2))))
       (fn [[A B]] (map #(apply str (map first (clojure.string/split % #" "))) (intersection #{"Cherry 1" "Apple Pink Lady" "Apple Red 1"} A)))))
