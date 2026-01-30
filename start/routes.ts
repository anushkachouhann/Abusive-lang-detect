/*
|--------------------------------------------------------------------------
| Routes file
|--------------------------------------------------------------------------
|
| The routes file is used for defining the HTTP routes.
|
*/

import router from '@adonisjs/core/services/router'
import app from '@adonisjs/core/services/app'
const ModerationsController = () => import('#controllers/moderations_controller')


router.get('/', async () => {
  return {
    hello: 'world',
  }
})

// Serve test HTML file
router.get('/test', async ({ response }) => {
  return response.download(app.makePath('test.html'))
})

// Serve OpenVINO test HTML file
router.get('/openvino-test', async ({ response }) => {
  return response.download(app.makePath('openvino-test.html'))
})

// NSFW Detection routes
router.group(() => {
  router.post('/analyze', '#controllers/moderations_controller.analyze')
  router.get('/results', '#controllers/moderations_controller.index')
  router.get('/results/:id', '#controllers/moderations_controller.show')
  router.delete('/results/:id', '#controllers/moderations_controller.destroy')
  router.get('/health', '#controllers/moderations_controller.health')
}).prefix('/api/moderation')

router.post('/api/analyze', [ModerationsController, 'analyze'])
router.get('/api/moderations', [ModerationsController, 'index'])
router.get('/api/moderations/:id', [ModerationsController, 'show'])
router.delete('/api/moderations/:id', [ModerationsController, 'destroy'])
router.get('/api/health', [ModerationsController, 'health'])
router.post('/api/test-openvino', [ModerationsController, 'testOpenVINO'])
// OpenVINO routes - using existing methods
router.post('/api/moderation/analyze-nsfw', [ModerationsController, 'analyzeNSFW'])