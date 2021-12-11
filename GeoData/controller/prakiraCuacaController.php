<?php
    require_once 'controller/service/view.php';

    class prakiraCuacaController {
        public function start() {
            return View::createView('prakiraCuaca.php', []);
        }
    }
?>