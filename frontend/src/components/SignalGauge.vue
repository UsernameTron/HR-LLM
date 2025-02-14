<template>
  <div class="signal-gauge">
    <v-card class="mx-auto" max-width="400">
      <v-card-title class="text-center">
        Hiring Signal Strength
      </v-card-title>
      <v-card-text>
        <canvas ref="gaugeCanvas"></canvas>
        <div class="text-center mt-4">
          <div class="text-h4">{{ signalStrength }}%</div>
          <div class="text-subtitle-1">{{ confidenceLevel }}</div>
        </div>
      </v-card-text>
    </v-card>
  </div>
</template>

<script>
import { Chart } from 'chart.js/auto'
import { defineComponent, onMounted, ref, watch } from 'vue'

export default defineComponent({
  name: 'SignalGauge',
  
  props: {
    value: {
      type: Number,
      required: true,
      default: 0
    },
    confidence: {
      type: String,
      required: true,
      default: 'Low'
    }
  },

  setup(props) {
    const gaugeCanvas = ref(null)
    const chart = ref(null)
    const signalStrength = ref(props.value)
    const confidenceLevel = ref(props.confidence)

    const createGauge = () => {
      const ctx = gaugeCanvas.value.getContext('2d')
      chart.value = new Chart(ctx, {
        type: 'doughnut',
        data: {
          datasets: [{
            data: [props.value, 100 - props.value],
            backgroundColor: [
              'rgba(75, 192, 192, 0.8)',
              'rgba(200, 200, 200, 0.2)'
            ],
            circumference: 180,
            rotation: 270
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          cutout: '80%',
          plugins: {
            legend: {
              display: false
            },
            tooltip: {
              enabled: false
            }
          }
        }
      })
    }

    watch(() => props.value, (newValue) => {
      signalStrength.value = newValue
      if (chart.value) {
        chart.value.data.datasets[0].data = [newValue, 100 - newValue]
        chart.value.update()
      }
    })

    watch(() => props.confidence, (newValue) => {
      confidenceLevel.value = newValue
    })

    onMounted(() => {
      createGauge()
    })

    return {
      gaugeCanvas,
      signalStrength,
      confidenceLevel
    }
  }
})
</script>

<style scoped>
.signal-gauge {
  height: 300px;
  position: relative;
}
canvas {
  height: 200px !important;
}
</style>
