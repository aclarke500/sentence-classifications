<template>
<h1>Hello world</h1>
<div class="container">
<div class="table-row header">
  <h2 class="sentence">Sentences</h2>
  <h2 class="class">ANN Class</h2>
  <h2 class="class">Euclidian Class</h2>
</div>
<div v-for="row in state.data" class="table-row">
  <p class="sentence">{{row.sentence}}</p>
  <p class="class">{{row.annClass}}</p>
  <p class="class">{{row.euclidianClass}}</p>
  </div>


</div>
</template>
<script setup>
import {reactive, ref} from 'vue'

const props = defineProps(['modelOutput'])
const state = reactive({
  data: []
})

createData();





function createData(){
  for (let i = 0; i < props.modelOutput.sentences.length; i++){
  state.data.push({
      sentence: props.modelOutput.sentences[i],
      annClass: props.modelOutput.ann[i],
      euclidianClass: props.modelOutput.euclidian[i]
    })
  }

}

</script>
<style scoped>
.sentence{
  flex:3;
}
.class{
  flex:2;
}
.container{
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;

}
.table-row{
  display: flex;
  flex-direction: row;
  justify-content: center;
  align-items: center;
  margin-bottom: 1rem;
  width: 100%;
  margin: 0;
}
.header{
  font-weight: bold;
}
</style>