<template>
  <Title id="title"></Title>
  <div class="container">
    <textarea v-model="state.input" placeholder="Enter some text"></textarea>
    <div class="button-bar">
      <button @click="analyzeText">Analyze</button>
    </div>
  </div>
  <Table v-if="state.modelOutput" :modelOutput="state.modelOutput" id="table"/>
</template>

<script setup>
import Title from '../components/Title.vue'
import Table from '../components/Table.vue'
import {reactive} from 'vue'

const state = reactive({
  input: 'Hello, how are you? WHat is your name?',
  modelOutput: null
})

async function analyzeText(){
  state.modelOutput = null;
    // Define the data you want to send
  const data = {
    text:state.input
  };

  // Use the fetch API to send the data to the Flask server
  fetch('http://127.0.0.1:5000/predict_classes', {
    method: 'POST', // or 'PUT'
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  })
  .then(response => response.json())
  .then(data => {
    console.log('Success:', data);
    state.modelOutput = data;
  })
  .catch((error) => {
    console.error('Error:', error);
  });
}


</script>
<style scoped>
.container{
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
}
#title{
  margin-bottom: 3rem;
}

textarea{
  width: 300px;
  height: 200px;
  margin-bottom: 20px;
}
.button-bar{
  display: flex;
  justify-content: center;
  align-items: center;
}

button{
  margin: 0 10px;
  padding: 10px 20px;
  background-color: #42b983;
  color: white;
  border: none;
  font-size: larger;
}
button:hover{
  background-color: #2c3e50;

}


</style>
